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
# Etapa 2 (METODOLOGIA_DT_HDT.md seccion 3): obs de pixeles + accion
# discreta. num_actions sale del cfg del snapshot, no de esta constante.
COINRUN_TASK = "coinrun"
COINRUN_OBS_SHAPE = (64, 64, 3)
TASKS = list(OBS_ACTION_DIMS) + [COINRUN_TASK]


def default_data_dir(task):
    if task == COINRUN_TASK:
        return Path("data") / COINRUN_TASK
    return Path("data") / f"{task}_medium_expert" / task


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


def token_modalities(model):
    """Modalidad de cada token dentro de un timestep, en orden: (return,
    state, action) para DT/HDT, (state, action) para BC sin rtg
    (use_rtg=False, METODOLOGIA_DT_HDT.md seccion 7). La posicion
    `idx` de la secuencia intercalada es el timestep idx // n y la
    modalidad modalities[idx % n], con n = len(modalities)."""
    if getattr(model, "use_rtg", True):
        return ("return", "state", "action")
    return ("state", "action")


def query_index(modalities, traj_length):
    """Posicion del ultimo token de estado, el que predice la accion final."""
    return len(modalities) * (traj_length - 1) + modalities.index("state")


def load_agent(snapshot, agent_kind, task, device):
    payload = torch.load(snapshot, map_location=device)
    if task == COINRUN_TASK:
        # mismo armado que eval_coinrun_seeds.py::load_agent
        obs_shape = COINRUN_OBS_SHAPE
        action_shape = (payload["cfg"].num_actions,)
    else:
        obs_dim, action_dim = OBS_ACTION_DIMS[task]
        obs_shape, action_shape = (obs_dim,), (action_dim,)
    agent_cls = DTAgent if agent_kind == "dt" else HDTAgent
    agent = agent_cls(
        name=f"{agent_kind}_attattr",
        obs_shape=obs_shape,
        action_shape=action_shape,
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
    tomada DESDE obs[i]). rtg[i] = return-to-go CRUDO (sin escalar) del
    episodio completo, sin descontar, en start_idx+i -- identico a
    `episode["rtg"]` del buffer con return_to_go=True, que es lo que ven
    los snapshots rtgfix en entrenamiento (METODOLOGIA seccion 2.10). Antes
    se recalculaba sobre la ventana y descontado, que es el rtg defectuoso
    de antes del fix (seccion 4.5)."""
    episode = load_episode(episode_path, task, None)
    assert start_idx >= 1, "start_idx debe ser >= 1 (idx-1 se usa para el estado inicial)"
    assert start_idx - 1 + traj_length <= episode_len(episode), (
        f"ventana [{start_idx - 1}, {start_idx - 1 + traj_length}) se sale del episodio "
        f"(largo {episode_len(episode)})"
    )
    rtg_full = np.cumsum(episode["reward"][::-1], axis=0)[::-1].astype(np.float32)
    obs = episode["observation"][start_idx - 1 : start_idx - 1 + traj_length]
    action = episode["action"][start_idx : start_idx + traj_length]
    rtg = rtg_full[start_idx : start_idx + traj_length]
    timestep = np.arange(start_idx - 1, start_idx - 1 + traj_length)
    return obs, action, rtg, timestep


def window_tensors(agent, obs, action, rtg, timestep, device):
    """(1, T, ...) tensores listos para agent.model. Obs de pixeles van en
    float32 (el encoder hace .float()/255 igual), para que SARFA pueda
    meter frames difuminados no enteros; acciones discretas en int64. rtg
    se escala por return_scale igual que en DTAgent.update_actor."""
    obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=device).unsqueeze(0)
    action_dtype = torch.long if agent.model.discrete_actions else torch.float32
    action_t = torch.as_tensor(np.asarray(action), dtype=action_dtype, device=device).unsqueeze(0)
    rtg_t = torch.as_tensor(np.asarray(rtg), dtype=torch.float32, device=device).unsqueeze(0)
    rtg_t = rtg_t / agent.return_scale
    timestep_t = torch.as_tensor(timestep, dtype=torch.long, device=device).unsqueeze(0)
    return obs_t, action_t, rtg_t, timestep_t


def resolve_target(agent, obs_t, action_t, rtg_t, timestep_t, target_dim):
    """Indice de salida a atribuir en a_pred[-1]. Accion continua: la
    dimension `target_dim` (seccion 4.1). Accion discreta (CoinRun): el
    logit de la accion argmax (la que el agente elegiria), igual que
    analysis/attn_attr.py de Benjamin (rama upstream/hier-procgen-attattr);
    `target_dim` se ignora."""
    if not agent.model.discrete_actions:
        return target_dim
    with torch.no_grad():
        logits = agent.model(rtg_t, obs_t, action_t, timesteps=timestep_t)[0, -1]
    return int(logits.argmax().item())


def att_attr(agent, obs, action, rtg, timestep, target_dim, m, device):
    """Devuelve un array (n_layer, n_head, 3T, 3T): Attr_l por capa del
    stack `agent.model.blocks`, para el objetivo escalar
    `a_pred[-1, target]` (salida en el ultimo timestep de la ventana;
    target segun resolve_target). Cada capa se atribuye por separado (las
    demas quedan con alpha=1, es decir sin intervenir), siguiendo el
    analisis por capa de Hao et al. 2021 -- ver seccion 4.1 del documento
    de metodologia."""
    model = agent.model
    obs_t, action_t, rtg, timestep_t = window_tensors(agent, obs, action, rtg, timestep, device)
    target_idx = resolve_target(agent, obs_t, action_t, rtg, timestep_t, target_dim)

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
            target = pred_a[0, -1, target_idx]
            (grad,) = torch.autograd.grad(target, capture[0])
            if step == m:
                att_alpha1 = capture[0].detach()
            avg_grad = grad.detach() if avg_grad is None else avg_grad + grad.detach()
        attn.forward = original_forward
        avg_grad = avg_grad / m
        attrs.append((att_alpha1 * avg_grad).squeeze(0).cpu().numpy())  # (n_head, 3T, 3T)

    return np.stack(attrs, axis=0)  # (n_layer, n_head, 3T, 3T)


def summarize(attr, traj_length, top_k=5, modalities=("return", "state", "action")):
    """Imprime, por capa, los (timestep, modalidad) pasados con mayor
    atribucion hacia la posicion de query que produce la prediccion final
    (ultimo token de estado, ver query_index). Suma sobre heads."""
    n = len(modalities)
    query_idx = query_index(modalities, traj_length)
    n_layer = attr.shape[0]
    for l in range(n_layer):
        row = np.abs(attr[l]).sum(axis=0)[query_idx]  # (3T,) sumado sobre heads
        order = np.argsort(-row)[:top_k]
        print(f"  capa {l}: top-{top_k} posiciones clave por |atribucion|")
        for key_idx in order:
            t = key_idx // n
            mod = modalities[key_idx % n]
            print(f"    t={t:2d} ({mod:6s})  |attr|={row[key_idx]:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True, help="Path al snapshot_*.pt")
    parser.add_argument("--agent", choices=["dt", "hdt"], required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
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

    obs, action, rtg, timestep = load_window(
        args.task, episode_path, start_idx, traj_length
    )
    attr = att_attr(
        agent, obs, action, rtg, timestep,
        args.target_dim, args.m, args.device,
    )

    print(
        f"{args.agent} {args.task} (dim={args.target_dim}, m={args.m}, "
        f"ventana=[{start_idx - 1}, {start_idx - 1 + traj_length}) de {episode_path.name}): "
        f"atribucion shape {attr.shape} (n_layer, n_head, 3T, 3T)"
    )
    summarize(attr, traj_length, modalities=token_modalities(agent.model))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, attr=attr, start_idx=start_idx, traj_length=traj_length)
        print(f"  guardado: {out_path}")


if __name__ == "__main__":
    main()
