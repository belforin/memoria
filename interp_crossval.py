"""
Agrega attattr.py/sarfa.py sobre varias ventanas (episodio x posicion de
inicio) para reproducir sistematicamente la validacion cruzada AttAttr vs.
SARFA de METODOLOGIA_DT_HDT.md, seccion 4.3 (hecha a mano la primera vez,
sobre el checkpoint de la primera tanda -- este script existe para poder
repetirla sobre cualquier snapshot, en particular los snapshots con el
return-to-go corregido, seccion 2.10/2.12).

Reporta, sobre las N=--n-episodes x --n-starts ventanas:
  - correlacion de Spearman por posicion, AttAttr (agregado sobre capas y
    heads) vs. SARFA -- media +/- std y rango.
  - correlacion de Spearman por CAPA individual de AttAttr (sin agregar
    entre capas) vs. SARFA, para descartar que agregar capas sea la causa
    de una eventual divergencia.
  - acuerdo en que modalidad (return/state/action) domina el score de cada
    metodo (suma de |score| sobre las posiciones de esa modalidad).

Corre en `maskdp-env`, igual que attattr.py/sarfa.py.
"""
import argparse
from pathlib import Path

import numpy as np

from attattr import OBS_ACTION_DIMS, att_attr, load_agent, load_window
from replay_buffer import episode_len, load_episode
from sarfa import attattr_per_position, sarfa

MODALITY_NAMES = {0: "return", 1: "state", 2: "action"}


def _spearman(a, b):
    # Igual que sarfa.py::cross_validate._spearman -- correlacion de Pearson
    # sobre los rangos, sin depender de scipy.
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _dominant_modality(score_per_position):
    totals = np.zeros(3)
    for idx, s in enumerate(score_per_position):
        totals[idx % 3] += abs(s)
    return int(np.argmax(totals))


def per_layer_scores(attr, traj_length):
    """(n_layer, n_candidates): |atribucion| sumada sobre heads, fila del
    token de consulta (query_idx), restringida a las posiciones candidatas
    0..query_idx -- a diferencia de attattr_per_position (sarfa.py), NO
    agrega entre capas."""
    query_idx = 3 * (traj_length - 1) + 1
    row = np.abs(attr).sum(axis=1)[:, query_idx, : query_idx + 1]  # (n_layer, n_candidatos)
    return row


def run_window(agent, task, episode_path, start_idx, target_dim, m, device, traj_length):
    obs, action, reward, discount, timestep = load_window(task, episode_path, start_idx, traj_length)
    attr = att_attr(agent, obs, action, reward, discount, timestep, target_dim, m, device)
    attattr_score = attattr_per_position(attr, traj_length)
    sarfa_score, _, _, _ = sarfa(agent, obs, action, reward, discount, timestep, target_dim, device)

    rho = _spearman(attattr_score, sarfa_score)
    layer_scores = per_layer_scores(attr, traj_length)
    per_layer_rho = [_spearman(layer_scores[l], sarfa_score) for l in range(layer_scores.shape[0])]

    return dict(
        rho=rho, per_layer_rho=per_layer_rho,
        dom_attattr=_dominant_modality(attattr_score),
        dom_sarfa=_dominant_modality(sarfa_score),
    )


def pick_start_indices(ep_len, traj_length, n_starts):
    """Posiciones de inicio equiespaciadas al 25/50/75% (generalizado a
    n_starts) del rango valido [1, ep_len-traj_length+1], deduplicadas."""
    max_start = ep_len - traj_length + 1
    assert max_start >= 1, f"episodio de largo {ep_len} no alcanza para traj_length={traj_length}"
    fracs = np.linspace(0.25, 0.75, n_starts) if n_starts > 1 else [0.5]
    starts = sorted({max(1, min(max_start, int(round(f * max_start)))) for f in fracs})
    i = 1
    while len(starts) < n_starts:
        candidate = min(max_start, starts[-1] + i)
        if candidate not in starts:
            starts.append(candidate)
        i += 1
    return sorted(starts)[:n_starts]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True, help="Path al snapshot_*.pt")
    parser.add_argument("--agent", choices=["dt", "hdt"], required=True)
    parser.add_argument("--task", choices=list(OBS_ACTION_DIMS), required=True)
    parser.add_argument("--data-dir", default=None,
                         help="Default: data/<task>_medium_expert/<task>")
    parser.add_argument("--n-episodes", type=int, default=3)
    parser.add_argument("--n-starts", type=int, default=3)
    parser.add_argument("--target-dim", type=int, default=0)
    parser.add_argument("--m", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=None, help="Path .npz donde guardar los resultados crudos")
    args = parser.parse_args()

    agent = load_agent(args.snapshot, args.agent, args.task, args.device)
    traj_length = agent.config.traj_length

    data_dir = Path(args.data_dir) if args.data_dir else Path("data") / f"{args.task}_medium_expert" / args.task
    episode_paths = sorted(data_dir.glob("episode_*.npz"))[: args.n_episodes]
    assert len(episode_paths) == args.n_episodes, (
        f"se pidieron {args.n_episodes} episodios, hay {len(episode_paths)} en {data_dir}"
    )

    print(f"{args.agent} {args.task} target_dim={args.target_dim} "
          f"({args.n_episodes} episodios x {args.n_starts} posiciones = "
          f"{args.n_episodes * args.n_starts} ventanas)\n")

    results = []
    for ep_path in episode_paths:
        ep_len = episode_len(load_episode(ep_path, args.task, None))
        for start_idx in pick_start_indices(ep_len, traj_length, args.n_starts):
            r = run_window(agent, args.task, ep_path, start_idx, args.target_dim, args.m, args.device, traj_length)
            r.update(episode=ep_path.name, start_idx=start_idx)
            results.append(r)
            print(
                f"{ep_path.name} start={start_idx:4d}: rho={r['rho']:+.3f}  "
                f"dominante AttAttr={MODALITY_NAMES[r['dom_attattr']]:6s} "
                f"SARFA={MODALITY_NAMES[r['dom_sarfa']]:6s}"
            )

    rhos = np.array([r["rho"] for r in results])
    agree = sum(r["dom_attattr"] == r["dom_sarfa"] for r in results)
    per_layer = np.array([r["per_layer_rho"] for r in results])  # (n_ventanas, n_layer)

    print(f"\n[agregado sobre {len(results)} ventanas]")
    print(f"  Spearman agregado (capas+heads) vs SARFA: media {rhos.mean():+.3f} "
          f"+/- {rhos.std():.3f}, rango [{rhos.min():+.3f}, {rhos.max():+.3f}]")
    for l in range(per_layer.shape[1]):
        col = per_layer[:, l]
        print(f"  Spearman capa {l} (sin agregar) vs SARFA: media {col.mean():+.3f} +/- {col.std():.3f}")
    print(f"  acuerdo en modalidad dominante: {agree}/{len(results)} ({100*agree/len(results):.0f}%)")

    dom_sarfa = np.bincount([r["dom_sarfa"] for r in results], minlength=3)
    dom_attattr = np.bincount([r["dom_attattr"] for r in results], minlength=3)
    print("  SARFA dominante por modalidad:   " + ", ".join(f"{MODALITY_NAMES[i]}={dom_sarfa[i]}" for i in range(3)))
    print("  AttAttr dominante por modalidad: " + ", ".join(f"{MODALITY_NAMES[i]}={dom_attattr[i]}" for i in range(3)))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path, rhos=rhos, per_layer=per_layer,
            dom_attattr=[r["dom_attattr"] for r in results],
            dom_sarfa=[r["dom_sarfa"] for r in results],
            episodes=[r["episode"] for r in results],
            start_idx=[r["start_idx"] for r in results],
        )
        print(f"  guardado: {out_path}")


if __name__ == "__main__":
    main()
