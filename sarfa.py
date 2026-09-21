"""
SARFA (Puri et al. 2020, "Explain Your Move") reformulado para accion
continua sobre DT/HDT. Ver METODOLOGIA_DT_HDT.md, seccion 4.1, para el
diseno completo y su justificacion -- a diferencia de AttAttr (attattr.py),
esto NO es una adaptacion mecanica del metodo original (definido sobre una
distribucion softmax de acciones discretas): specificity/relevance se
redefinen desde cero para un vector de accion continuo.

Unidad de perturbacion: un token completo de la ventana de contexto
(R_t, s_t o a_t, misma intercalacion que agent/dt.py::DecisionTransformer.
forward), reemplazado por su "valor neutro":
  - estado (s_t): el buffer model.obs_mean (asi que, tras la normalizacion
    z-score interna del modelo, el valor efectivo que ve la red es
    exactamente 0 -- ver METODOLOGIA seccion 2.8/4.1).
  - return-to-go (R_t) y accion (a_t): 0.0 directo (no hay normalizacion
    de por medio para estas dos modalidades en el pipeline).

Reusa OBS_ACTION_DIMS/load_agent/load_window de attattr.py -- mismo
contrato de datos (ventana real de un episodio ya convertido,
misma convencion de alineacion que OfflineReplayBuffer._sample), para que
ambos metodos se puedan correr sobre exactamente la misma ventana y
cruzarse (ver cross_validate() mas abajo).

Corre en `maskdp-env` (solo necesita torch + numpy), igual que attattr.py.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from agent.dt import DTAgent
from attattr import OBS_ACTION_DIMS, att_attr, load_agent, load_window

EPS = 1e-8


def _forward_action(model, rtg, obs, action, timesteps):
    with torch.no_grad():
        pred_a = model(rtg, obs, action, timesteps=timesteps)
    return pred_a[0, -1].cpu().numpy()  # (action_dim,) -- prediccion en el ultimo timestep


def sarfa(agent, obs, action, reward, discount, timestep, target_dim, device):
    """
    Devuelve (sarfa_score, specificity, relevance, delta_a), cada uno un
    array (n_candidates,) salvo delta_a que es (n_candidates, action_dim),
    para las posiciones candidatas r = 0..query_idx (inclusive) de la
    secuencia intercalada (R_0,s_0,a_0,...,R_{T-1},s_{T-1}) -- el token
    a_{T-1} (la propia prediccion objetivo) queda excluido: la mascara
    causal ya lo bloquea de influir sobre si mismo, perturbarlo no tiene
    sentido. `target_dim` es la dimension de accion `j` de interes (ver
    METODOLOGIA seccion 4.1).
    """
    model = agent.model
    T = obs.shape[0]
    query_idx = 3 * (T - 1) + 1  # posicion del token s_{T-1} en la secuencia intercalada
    n_candidates = query_idx + 1

    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    action_t = torch.as_tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
    reward_t = torch.as_tensor(reward, dtype=torch.float32, device=device).unsqueeze(0)
    discount_t = torch.as_tensor(discount, dtype=torch.float32, device=device).unsqueeze(0)
    timestep_t = torch.as_tensor(timestep, dtype=torch.long, device=device).unsqueeze(0)

    rtg = DTAgent.compute_returns_to_go(reward_t, discount_t) / agent.return_scale

    a_ref = _forward_action(model, rtg, obs_t, action_t, timestep_t)

    action_dim = a_ref.shape[0]
    delta_a = np.zeros((n_candidates, action_dim), dtype=np.float64)
    for idx in range(n_candidates):
        t, mod = idx // 3, idx % 3  # mod: 0=return, 1=state, 2=action

        obs_p, action_p, rtg_p = obs_t.clone(), action_t.clone(), rtg.clone()
        if mod == 0:
            rtg_p[0, t] = 0.0
        elif mod == 1:
            # ablar a la media (buffer obs_mean): tras la normalizacion
            # z-score interna del modelo esto equivale a "estado promedio",
            # es decir 0 en el espacio normalizado -- ver docstring.
            obs_p[0, t] = model.obs_mean
        else:
            action_p[0, t] = 0.0

        a_pert = _forward_action(model, rtg_p, obs_p, action_p, timestep_t)
        delta_a[idx] = a_pert - a_ref

    dj = np.abs(delta_a[:, target_dim])
    specificity = (dj - dj.min()) / (dj.max() - dj.min() + EPS)

    l1_total = np.abs(delta_a).sum(axis=1)
    relevance = dj / (l1_total + EPS)

    sarfa_score = 2 * specificity * relevance / (specificity + relevance + EPS)
    return sarfa_score, specificity, relevance, delta_a


MODALITY_NAMES = {0: "return", 1: "state", 2: "action"}


def group_ablation(agent, obs, action, reward, discount, timestep, target_dim, device,
                    delta_a=None):
    """
    Prueba la hipotesis de redundancia entre tokens vecinos de una misma
    modalidad (METODOLOGIA_DT_HDT.md seccion 4.3, "en curso"): en vez de
    ablacionar un token a la vez (sarfa()), ablaciona TODOS los tokens de
    una modalidad a la vez dentro de la ventana, y compara |Δa_j| del
    efecto conjunto contra la suma/maximo de los efectos individuales ya
    medidos por sarfa(). Si una modalidad es redundante (cada token solo
    aporta poco porque sus vecinos cargan info casi equivalente, pero
    juntos sí importan), el efecto de grupo deberia ser
    desproporcionadamente mayor que la suma de los efectos individuales.

    `delta_a`: si ya se corrio sarfa() sobre esta misma ventana/target, se
    puede pasar su delta_a para no recalcular los efectos individuales.

    Devuelve un dict {modalidad: {"group": float, "sum_individual": float,
    "max_individual": float, "n_tokens": int}}.
    """
    model = agent.model
    T = obs.shape[0]
    query_idx = 3 * (T - 1) + 1

    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    action_t = torch.as_tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
    reward_t = torch.as_tensor(reward, dtype=torch.float32, device=device).unsqueeze(0)
    discount_t = torch.as_tensor(discount, dtype=torch.float32, device=device).unsqueeze(0)
    timestep_t = torch.as_tensor(timestep, dtype=torch.long, device=device).unsqueeze(0)

    rtg = DTAgent.compute_returns_to_go(reward_t, discount_t) / agent.return_scale
    a_ref = _forward_action(model, rtg, obs_t, action_t, timestep_t)

    if delta_a is None:
        _, _, _, delta_a = sarfa(agent, obs, action, reward, discount, timestep, target_dim, device)

    results = {}
    for mod in range(3):
        positions = [idx for idx in range(query_idx + 1) if idx % 3 == mod]
        t_positions = [idx // 3 for idx in positions]

        obs_p, action_p, rtg_p = obs_t.clone(), action_t.clone(), rtg.clone()
        if mod == 0:
            for t in t_positions:
                rtg_p[0, t] = 0.0
        elif mod == 1:
            for t in t_positions:
                obs_p[0, t] = model.obs_mean
        else:
            for t in t_positions:
                action_p[0, t] = 0.0

        a_pert_group = _forward_action(model, rtg_p, obs_p, action_p, timestep_t)
        group_j = float(abs(a_pert_group[target_dim] - a_ref[target_dim]))

        individual_js = np.abs(delta_a[positions, target_dim])
        results[MODALITY_NAMES[mod]] = dict(
            group=group_j,
            sum_individual=float(individual_js.sum()),
            max_individual=float(individual_js.max()),
            n_tokens=len(positions),
        )
    return results


def summarize(sarfa_score, traj_length, top_k=5):
    """Mismo formato de salida que attattr.py::summarize, para poder
    comparar a simple vista."""
    modality = {0: "return", 1: "state", 2: "action"}
    order = np.argsort(-sarfa_score)[:top_k]
    print(f"  top-{top_k} posiciones clave por SARFA_j")
    for idx in order:
        t, mod = idx // 3, idx % 3
        print(f"    t={t:2d} ({modality[mod]:6s})  SARFA={sarfa_score[idx]:.6f}")


def attattr_per_position(attr, traj_length):
    """Colapsa la atribucion de AttAttr (n_layer, n_head, 3T, 3T) a un
    score escalar por posicion candidata, comparable a sarfa_score: suma
    |atribucion| sobre heads y capas, fila del token de consulta
    (query_idx), restringido a las mismas posiciones candidatas que
    sarfa() (0..query_idx inclusive)."""
    query_idx = 3 * (traj_length - 1) + 1
    # (n_layer, n_head, 3T, 3T) -> sumar |.| sobre capas y heads -> (3T, 3T)
    row = np.abs(attr).sum(axis=(0, 1))[query_idx]  # (3T,)
    return row[: query_idx + 1]


def cross_validate(agent, obs, action, reward, discount, timestep, target_dim, m, device,
                    traj_length, top_k=5):
    """
    Corre AttAttr y SARFA sobre la MISMA ventana/checkpoint y compara
    cualitativamente que timesteps pasados marcan como importantes --
    paso pedido explicitamente en METODOLOGIA_DT_HDT.md seccion 4.1 antes
    de confiar en cualquiera de los dos metodos para conclusiones DT vs.
    HDT. Reporta correlacion de Spearman entre ambos scores por posicion
    y overlap de sus respectivos top-k.
    """
    attr = att_attr(agent, obs, action, reward, discount, timestep, target_dim, m, device)
    attattr_score = attattr_per_position(attr, traj_length)

    sarfa_score, _, _, _ = sarfa(agent, obs, action, reward, discount, timestep, target_dim, device)

    assert attattr_score.shape == sarfa_score.shape, (
        f"shapes no calzan: attattr {attattr_score.shape} vs sarfa {sarfa_score.shape}"
    )

    # Spearman sin depender de scipy: correlacion de Pearson sobre los rangos.
    def _spearman(a, b):
        ra = np.argsort(np.argsort(a))
        rb = np.argsort(np.argsort(b))
        if ra.std() == 0 or rb.std() == 0:
            return float("nan")
        return float(np.corrcoef(ra, rb)[0, 1])

    rho = _spearman(attattr_score, sarfa_score)

    top_attattr = set(np.argsort(-attattr_score)[:top_k].tolist())
    top_sarfa = set(np.argsort(-sarfa_score)[:top_k].tolist())
    overlap = len(top_attattr & top_sarfa)

    modality = {0: "return", 1: "state", 2: "action"}
    print(f"\n[cross-validacion] correlacion de Spearman (AttAttr vs SARFA): {rho:.3f}")
    print(f"[cross-validacion] overlap top-{top_k}: {overlap}/{top_k}")
    print(f"  AttAttr top-{top_k}: " + ", ".join(
        f"t={i // 3}({modality[i % 3]})" for i in sorted(top_attattr, key=lambda i: -attattr_score[i])
    ))
    print(f"  SARFA   top-{top_k}: " + ", ".join(
        f"t={i // 3}({modality[i % 3]})" for i in sorted(top_sarfa, key=lambda i: -sarfa_score[i])
    ))
    return rho, overlap


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
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=None, help="Path .npz donde guardar los scores completos")
    parser.add_argument("--cross-validate", action="store_true",
                         help="Tambien correr AttAttr sobre la misma ventana y comparar (seccion 4.1)")
    parser.add_argument("--m", type=int, default=20, help="Pasos de integracion de AttAttr (solo con --cross-validate)")
    args = parser.parse_args()

    agent = load_agent(args.snapshot, args.agent, args.task, args.device)
    traj_length = agent.config.traj_length

    from replay_buffer import episode_len, load_episode
    episode_path = Path(args.episode)
    ep_len = episode_len(load_episode(episode_path, args.task, None))
    start_idx = args.start_idx if args.start_idx is not None else max(1, ep_len // 2)

    obs, action, reward, discount, timestep = load_window(
        args.task, episode_path, start_idx, traj_length
    )

    sarfa_score, specificity, relevance, delta_a = sarfa(
        agent, obs, action, reward, discount, timestep, args.target_dim, args.device
    )

    print(
        f"{args.agent} {args.task} (dim={args.target_dim}, "
        f"ventana=[{start_idx - 1}, {start_idx - 1 + traj_length}) de {episode_path.name}): "
        f"SARFA shape {sarfa_score.shape} ({len(sarfa_score)} candidatos)"
    )
    summarize(sarfa_score, traj_length)

    if args.cross_validate:
        cross_validate(
            agent, obs, action, reward, discount, timestep, args.target_dim, args.m,
            args.device, traj_length,
        )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path, sarfa=sarfa_score, specificity=specificity, relevance=relevance,
            delta_a=delta_a, start_idx=start_idx, traj_length=traj_length,
        )
        print(f"  guardado: {out_path}")


if __name__ == "__main__":
    main()
