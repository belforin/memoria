"""
AttAttr (Hao et al. 2021, "Self-Attention Attribution") adaptado a accion
continua para DT/HDT. Ver METODOLOGIA_DT_HDT.md, seccion 4.1, para el
diseno completo y su justificacion.

Calcula, para una ventana de contexto real tomada de un episodio ya
convertido (data/<task>_medium_expert/<task>/episode_*.npz), la
atribucion por integral de gradientes sobre los pesos de atencion de cada
capa del stack `agent.model.blocks` -- el unico tramo comparable entre
DT y HDT (stack completo en DT, stack superior post-fusion en HDT, ver
seccion 4 del documento) -- usando como objetivo escalar la accion
predicha en la dimension `--target-dim`, en el ultimo timestep de la
ventana (la prediccion "de hoy").

Corre en `maskdp-env` (solo necesita torch + numpy, no gym/mujoco_py):
no depende de eval_dt.py ni de un rollout en vivo, usa directamente una
ventana real del dataset offline con la misma convencion de alineacion
que `OfflineReplayBuffer._sample` (replay_buffer.py).

No modifica agent/modules/attention.py (compartido por todos los agentes
del repo, incluidos jobs de entrenamiento que puedan estar corriendo en
paralelo): en su lugar parchea el metodo `forward` de CausalSelfAttention
en tiempo de ejecucion, solo sobre la instancia ya cargada en este
proceso -- cero cambios al codigo de entrenamiento.
"""
import argparse
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from agent.dt import DTAgent
from agent.hdt import HDTAgent
from replay_buffer import episode_len, load_episode

OBS_ACTION_DIMS = {
    "halfcheetah": (17, 6),
    "hopper": (11, 3),
    "walker2d": (17, 6),
}


def _patched_attn_forward(self, x, mask, alpha=1.0, capture=None):
    """Identico a CausalSelfAttention.forward (agent/modules/attention.py),
    salvo que escala la matriz de atencion por `alpha` antes de aplicarla
    a `v`, y opcionalmente guarda la matriz sin escalar en `capture`
    (lista de un elemento, se le hace .append). Con esto,
    autograd.grad(F, capture[0]) devuelve exactamente ∂F(alpha*A)/∂A por
    regla de la cadena (A solo se usa via alpha*A en el resto del grafo),
    que es el termino que integra Hao et al. 2021, seccion 4.1 del
    documento de metodologia."""
    B, T, C = x.size()
    k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

    att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1) ** 0.5))
    att = att.masked_fill(mask[:, :, :T, :T] == 0, float("-inf"))
    att = F.softmax(att, dim=-1)
    att = self.attn_drop(att)
    if capture is not None:
        capture.append(att)

    y = (alpha * att) @ v
    y = y.transpose(1, 2).contiguous().view(B, T, C)
    y = self.resid_drop(self.proj(y))
    return y


def load_agent(snapshot, agent_kind, task, device):
    obs_dim, action_dim = OBS_ACTION_DIMS[task]
    payload = torch.load(snapshot, map_location=device)
    agent_cls = DTAgent if agent_kind == "dt" else HDTAgent
    agent = agent_cls(
        name=f"{agent_kind}_attattr",
        obs_shape=(obs_dim,),
        action_shape=(action_dim,),
        device=device,
        lr=1e-4,
        batch_size=1,
        stddev_schedule=0.2,
        use_tb=False,
        transformer_cfg=payload["cfg"],
    )
    # strict=False: checkpoints previos a la normalizacion de obs
    # (METODOLOGIA_DT_HDT.md seccion 2.8) no tienen obs_mean/obs_std.
    agent.model.load_state_dict(payload["model"], strict=False)
    agent.train(False)
    return agent


def load_window(task, episode_path, start_idx, traj_length):
    """Misma convencion de alineacion que OfflineReplayBuffer._sample
    (replay_buffer.py): obs[i] = observation[start_idx-1+i] (estado en el
    paso i de la ventana), action[i] = action[start_idx+i] (accion
    tomada DESDE obs[i]). reward/discount van desfasados igual que en
    _sample (alineados con la accion, no con el estado)."""
    episode = load_episode(episode_path, task, None)
    assert start_idx >= 1, "start_idx debe ser >= 1 (idx-1 se usa para el estado inicial)"
    assert start_idx - 1 + traj_length <= episode_len(episode), (
        f"ventana [{start_idx - 1}, {start_idx - 1 + traj_length}) se sale del episodio "
        f"(largo {episode_len(episode)})"
    )
    obs = episode["observation"][start_idx - 1 : start_idx - 1 + traj_length]
    action = episode["action"][start_idx : start_idx + traj_length]
    reward = episode["reward"][start_idx : start_idx + traj_length]
    discount = episode["discount"][start_idx : start_idx + traj_length]
    timestep = np.arange(start_idx - 1, start_idx - 1 + traj_length)
    return obs, action, reward, discount, timestep


def att_attr(agent, obs, action, reward, discount, timestep, target_dim, m, device):
    """Devuelve un array (n_layer, n_head, 3T, 3T): Attr_l por capa del
    stack `agent.model.blocks`, para el objetivo escalar
    `a_pred[-1, target_dim]` (accion predicha en el ultimo timestep de la
    ventana). Cada capa se atribuye por separado (las demas quedan con
    alpha=1, es decir sin intervenir), siguiendo el analisis por capa de
    Hao et al. 2021 -- ver seccion 4.1 del documento de metodologia."""
    model = agent.model
    T = obs.shape[0]

    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    action_t = torch.as_tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
    reward_t = torch.as_tensor(reward, dtype=torch.float32, device=device).unsqueeze(0)
    discount_t = torch.as_tensor(discount, dtype=torch.float32, device=device).unsqueeze(0)
    timestep_t = torch.as_tensor(timestep, dtype=torch.long, device=device).unsqueeze(0)

    rtg = DTAgent.compute_returns_to_go(reward_t, discount_t) / agent.return_scale

    attrs = []
    for layer in model.blocks:
        attn = layer.attn
        original_forward = attn.forward
        avg_grad = None
        att_alpha1 = None
        for step in range(1, m + 1):
            alpha = step / m
            capture = []

            def bound(self, x, mask, _alpha=alpha, _capture=capture):
                return _patched_attn_forward(self, x, mask, alpha=_alpha, capture=_capture)

            attn.forward = types.MethodType(bound, attn)
            pred_a = model(rtg, obs_t, action_t, timesteps=timestep_t)
            target = pred_a[0, -1, target_dim]
            (grad,) = torch.autograd.grad(target, capture[0])
            if step == m:
                att_alpha1 = capture[0].detach()
            avg_grad = grad.detach() if avg_grad is None else avg_grad + grad.detach()
        attn.forward = original_forward
        avg_grad = avg_grad / m
        attrs.append((att_alpha1 * avg_grad).squeeze(0).cpu().numpy())  # (n_head, 3T, 3T)

    return np.stack(attrs, axis=0)  # (n_layer, n_head, 3T, 3T)


def summarize(attr, traj_length, top_k=5):
    """Imprime, por capa, los (timestep, modalidad) pasados con mayor
    atribucion hacia la posicion de query que produce la prediccion final
    (ultimo token de estado, indice 3*(T-1)+1). Suma sobre heads."""
    modality = {0: "return", 1: "state", 2: "action"}
    query_idx = 3 * (traj_length - 1) + 1
    n_layer = attr.shape[0]
    for l in range(n_layer):
        row = np.abs(attr[l]).sum(axis=0)[query_idx]  # (3T,) sumado sobre heads
        order = np.argsort(-row)[:top_k]
        print(f"  capa {l}: top-{top_k} posiciones clave por |atribucion|")
        for key_idx in order:
            t = key_idx // 3
            mod = modality[key_idx % 3]
            print(f"    t={t:2d} ({mod:6s})  |attr|={row[key_idx]:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True, help="Path al snapshot_*.pt")
    parser.add_argument("--agent", choices=["dt", "hdt"], required=True)
    parser.add_argument("--task", choices=list(OBS_ACTION_DIMS), required=True)
    parser.add_argument("--episode", required=True, help="Path a un episode_*.npz ya convertido")
    parser.add_argument("--start-idx", type=int, default=None,
                         help="Indice inicial de la ventana (default: mitad del episodio)")
    parser.add_argument("--target-dim", type=int, default=0,
                         help="Dimension de accion a atribuir (ver METODOLOGIA seccion 4.1)")
    parser.add_argument("--m", type=int, default=20, help="Pasos de integracion (Hao et al. 2021)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=None, help="Path .npz donde guardar el tensor de atribucion completo")
    args = parser.parse_args()

    agent = load_agent(args.snapshot, args.agent, args.task, args.device)
    traj_length = agent.config.traj_length

    episode_path = Path(args.episode)
    ep_len = episode_len(load_episode(episode_path, args.task, None))
    start_idx = args.start_idx if args.start_idx is not None else max(1, ep_len // 2)

    obs, action, reward, discount, timestep = load_window(
        args.task, episode_path, start_idx, traj_length
    )
    attr = att_attr(
        agent, obs, action, reward, discount, timestep,
        args.target_dim, args.m, args.device,
    )

    print(
        f"{args.agent} {args.task} (dim={args.target_dim}, m={args.m}, "
        f"ventana=[{start_idx - 1}, {start_idx - 1 + traj_length}) de {episode_path.name}): "
        f"atribucion shape {attr.shape} (n_layer, n_head, 3T, 3T)"
    )
    summarize(attr, traj_length)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, attr=attr, start_idx=start_idx, traj_length=traj_length)
        print(f"  guardado: {out_path}")


if __name__ == "__main__":
    main()
