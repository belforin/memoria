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

Reusa load_agent/load_window/window_tensors de attattr.py -- mismo
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
from scipy.ndimage import gaussian_filter

from attattr import TASKS, att_attr, load_agent, load_window, resolve_target, window_tensors
from third_party.sarfa_saliency import computeSaliencyUsingSarfa

EPS = 1e-8

# CoinRun (accion discreta, pixeles): valores neutros de cada modalidad.
# - estado: frame completo difuminado con blur gaussiano sigma=3, igual que
#   la saliencia temporal de Benjamin (eval_sarfa.py::compute_state_saliency,
#   temporal_blur_sigma=null -> perturbation.blur_sigma=3.0, rama
#   upstream/hier-procgen-sarfa).
# - accion: indice 4, el NOOP de Procgen (no hay "accion 0" neutra en un
#   espacio discreto; 0 es LEFT+DOWN).
PIXEL_BLUR_SIGMA = 3.0
PROCGEN_NOOP = 4


def _forward_last(model, rtg, obs, action, timesteps):
    with torch.no_grad():
        pred_a = model(rtg, obs, action, timesteps=timesteps)
    return pred_a[0, -1].cpu().numpy()  # (action_dim,) o (num_actions,) logits -- ultimo timestep


def _ablate(model, obs_p, action_p, rtg_p, mod, t):
    """Reemplaza in-place el token (t, mod) por su valor neutro -- ver
    docstring del modulo y constantes de CoinRun arriba."""
    if mod == 0:
        rtg_p[0, t] = 0.0
    elif mod == 1:
        if model.pixel_obs:
            frame = obs_p[0, t].cpu().numpy()
            blurred = gaussian_filter(frame, sigma=(PIXEL_BLUR_SIGMA, PIXEL_BLUR_SIGMA, 0))
            obs_p[0, t] = torch.as_tensor(blurred, dtype=obs_p.dtype, device=obs_p.device)
        else:
            # ablar a la media (buffer obs_mean): tras la normalizacion
            # z-score interna del modelo esto equivale a "estado promedio",
            # es decir 0 en el espacio normalizado -- ver docstring.
            obs_p[0, t] = model.obs_mean
    else:
        action_p[0, t] = PROCGEN_NOOP if model.discrete_actions else 0.0


def sarfa(agent, obs, action, rtg, timestep, target_dim, device):
    """
    Devuelve (sarfa_score, specificity, relevance, delta), cada uno un
    array (n_candidates,) salvo delta que es (n_candidates, n_out), para
    las posiciones candidatas r = 0..query_idx (inclusive) de la
    secuencia intercalada (R_0,s_0,a_0,...,R_{T-1},s_{T-1}) -- el token
    a_{T-1} (la propia prediccion objetivo) queda excluido: la mascara
    causal ya lo bloquea de influir sobre si mismo, perturbarlo no tiene
    sentido.

    Accion continua: `target_dim` es la dimension de accion `j` de interes
    y specificity/relevance son la reformulacion de la seccion 4.1.
    Accion discreta (CoinRun): SARFA original de Puri et al. 2020 sobre los
    logits (third_party/sarfa_saliency.py, el mismo vendorizado por
    Benjamin), para la accion argmax; specificity = dP y relevance = K de
    esa implementacion, `target_dim` se ignora. `delta` son los cambios en
    la salida cruda (accion continua o logits).
    """
    model = agent.model
    obs_t, action_t, rtg_t, timestep_t = window_tensors(agent, obs, action, rtg, timestep, device)
    T = obs_t.shape[1]
    query_idx = 3 * (T - 1) + 1  # posicion del token s_{T-1} en la secuencia intercalada
    n_candidates = query_idx + 1

    out_ref = _forward_last(model, rtg_t, obs_t, action_t, timestep_t)
    a_hat = resolve_target(agent, obs_t, action_t, rtg_t, timestep_t, target_dim)

    delta = np.zeros((n_candidates, out_ref.shape[0]), dtype=np.float64)
    sarfa_score = np.zeros(n_candidates)
    specificity = np.zeros(n_candidates)
    relevance = np.zeros(n_candidates)
    dict_before = {k: float(v) for k, v in enumerate(out_ref)}
    for idx in range(n_candidates):
        t, mod = idx // 3, idx % 3  # mod: 0=return, 1=state, 2=action
        obs_p, action_p, rtg_p = obs_t.clone(), action_t.clone(), rtg_t.clone()
        _ablate(model, obs_p, action_p, rtg_p, mod, t)
        out_pert = _forward_last(model, rtg_p, obs_p, action_p, timestep_t)
        delta[idx] = out_pert - out_ref
        if model.discrete_actions:
            dict_after = {k: float(v) for k, v in enumerate(out_pert)}
            answer, dP, K, _, _, _ = computeSaliencyUsingSarfa(a_hat, dict_before, dict_after)
            sarfa_score[idx], specificity[idx], relevance[idx] = answer, dP, K

    if not model.discrete_actions:
        dj = np.abs(delta[:, a_hat])
        specificity = (dj - dj.min()) / (dj.max() - dj.min() + EPS)
        l1_total = np.abs(delta).sum(axis=1)
        relevance = dj / (l1_total + EPS)
        sarfa_score = 2 * specificity * relevance / (specificity + relevance + EPS)
    return sarfa_score, specificity, relevance, delta


MODALITY_NAMES = {0: "return", 1: "state", 2: "action"}


def group_ablation(agent, obs, action, rtg, timestep, target_dim, device, delta_a=None):
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
    En accion discreta, j es el logit de la accion argmax.

    `delta_a`: si ya se corrio sarfa() sobre esta misma ventana/target, se
    puede pasar su delta para no recalcular los efectos individuales.

    Devuelve un dict {modalidad: {"group": float, "sum_individual": float,
    "max_individual": float, "n_tokens": int}}.
    """
    model = agent.model
    obs_t, action_t, rtg_t, timestep_t = window_tensors(agent, obs, action, rtg, timestep, device)
    T = obs_t.shape[1]
    query_idx = 3 * (T - 1) + 1

    out_ref = _forward_last(model, rtg_t, obs_t, action_t, timestep_t)
    j = resolve_target(agent, obs_t, action_t, rtg_t, timestep_t, target_dim)

    if delta_a is None:
        _, _, _, delta_a = sarfa(agent, obs, action, rtg, timestep, target_dim, device)

    results = {}
    for mod in range(3):
        positions = [idx for idx in range(query_idx + 1) if idx % 3 == mod]

        obs_p, action_p, rtg_p = obs_t.clone(), action_t.clone(), rtg_t.clone()
        for idx in positions:
            _ablate(model, obs_p, action_p, rtg_p, mod, idx // 3)

        out_group = _forward_last(model, rtg_p, obs_p, action_p, timestep_t)
        group_j = float(abs(out_group[j] - out_ref[j]))

        individual_js = np.abs(delta_a[positions, j])
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


def cross_validate(agent, obs, action, rtg, timestep, target_dim, m, device,
                    traj_length, top_k=5):
    """
    Corre AttAttr y SARFA sobre la MISMA ventana/checkpoint y compara
    cualitativamente que timesteps pasados marcan como importantes --
    paso pedido explicitamente en METODOLOGIA_DT_HDT.md seccion 4.1 antes
    de confiar en cualquiera de los dos metodos para conclusiones DT vs.
    HDT. Reporta correlacion de Spearman entre ambos scores por posicion
    y overlap de sus respectivos top-k.
    """
    attr = att_attr(agent, obs, action, rtg, timestep, target_dim, m, device)
    attattr_score = attattr_per_position(attr, traj_length)

    sarfa_score, _, _, _ = sarfa(agent, obs, action, rtg, timestep, target_dim, device)

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
    parser.add_argument("--task", choices=TASKS, required=True)
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

    obs, action, rtg, timestep = load_window(
        args.task, episode_path, start_idx, traj_length
    )

    sarfa_score, specificity, relevance, delta_a = sarfa(
        agent, obs, action, rtg, timestep, args.target_dim, args.device
    )

    print(
        f"{args.agent} {args.task} (dim={args.target_dim}, "
        f"ventana=[{start_idx - 1}, {start_idx - 1 + traj_length}) de {episode_path.name}): "
        f"SARFA shape {sarfa_score.shape} ({len(sarfa_score)} candidatos)"
    )
    summarize(sarfa_score, traj_length)

    if args.cross_validate:
        cross_validate(
            agent, obs, action, rtg, timestep, args.target_dim, args.m,
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
