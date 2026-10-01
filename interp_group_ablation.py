"""
Ablacion por grupos (sarfa.py::group_ablation) agregada sobre varias
ventanas, para probar en las 4 tareas la hipotesis de redundancia de
METODOLOGIA_DT_HDT.md seccion 4.3: si ablacionar un solo token de una
modalidad casi no cambia la prediccion pero ablacionar todos los de esa
modalidad a la vez si, la modalidad es redundante (sus tokens cargan
informacion equivalente), no irrelevante.

Usa exactamente las mismas ventanas que interp_crossval.py (mismos
episodios y posiciones de inicio) para poder cruzar ambos resultados.

Reporta por modalidad, sobre las N ventanas (mediana y media):
  - group: |Δ salida_j| al ablacionar todos los tokens de la modalidad
    (accion continua: dimension j; CoinRun: |ΔP(a_hat)|, a_hat = argmax).
  - max_individual: el mayor |Δ salida_j| de un token solo.
  - ratio group/max_individual: >>1 = redundancia (juntos pesan mucho mas
    que cualquiera por separado); ~1 = un solo token carga el efecto.
  - group_l2: norma L2 del cambio en toda la salida (todas las dimensiones
    de accion, o todos los logits).
  - (CoinRun) fraccion de ventanas donde la accion argmax cambia.

Corre en `maskdp-env`, igual que interp_crossval.py.
"""
import argparse
from pathlib import Path

import numpy as np

from attattr import TASKS, default_data_dir, load_agent, load_window
from interp_crossval import pick_start_indices
from replay_buffer import episode_len, load_episode
from attattr import token_modalities
from sarfa import group_ablation

FIELDS = ("group", "sum_individual", "max_individual", "group_l2")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True, help="Path al snapshot_*.pt")
    parser.add_argument("--agent", choices=["dt", "hdt"], required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--min-len", type=int, default=None,
                         help="Igual que interp_crossval.py (60 para coinrun)")
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--n-starts", type=int, default=3)
    parser.add_argument("--target-dim", type=int, default=0,
                         help="Dimension de accion (continua); en coinrun se usa el logit argmax")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=None, help="Path .npz donde guardar los resultados crudos")
    args = parser.parse_args()

    agent = load_agent(args.snapshot, args.agent, args.task, args.device)
    traj_length = agent.config.traj_length

    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir(args.task)
    min_len = args.min_len if args.min_len is not None else traj_length
    episode_paths = [
        f for f in sorted(data_dir.glob("episode_*.npz")) if int(f.stem.split("_")[-1]) >= min_len
    ][: args.n_episodes]
    assert len(episode_paths) == args.n_episodes

    print(f"{args.agent} {args.task} ({args.n_episodes} episodios x {args.n_starts} posiciones)\n")

    rows = {m: {f: [] for f in FIELDS + ("argmax_changed",)} for m in token_modalities(agent.model)}
    for ep_path in episode_paths:
        ep_len = episode_len(load_episode(ep_path, args.task, None))
        for start_idx in pick_start_indices(ep_len, traj_length, args.n_starts):
            window = load_window(args.task, ep_path, start_idx, traj_length)
            res = group_ablation(agent, *window, args.target_dim, args.device)
            for mod, r in res.items():
                for f in FIELDS:
                    rows[mod][f].append(r[f])
                rows[mod]["argmax_changed"].append(r["argmax_changed"])
            print(f"{ep_path.name} start={start_idx:4d}: " + "  ".join(
                f"{mod}: grupo={r['group']:.4f} max_ind={r['max_individual']:.4f}"
                for mod, r in res.items()
            ))

    n = len(rows["state"]["group"])
    print(f"\n[agregado sobre {n} ventanas] mediana (media)")
    print(f"  {'modalidad':<9} {'grupo':>16} {'max individual':>16} {'grupo/max_ind':>16} {'grupo L2':>16}"
          + ("  argmax cambia" if agent.model.discrete_actions else ""))
    for mod, r in rows.items():
        g, mx, l2 = (np.array(r[f]) for f in ("group", "max_individual", "group_l2"))
        ratio = g / (mx + 1e-8)
        line = (f"  {mod:<9} {np.median(g):>7.4f} ({g.mean():.4f}) {np.median(mx):>7.4f} ({mx.mean():.4f}) "
                f"{np.median(ratio):>7.2f} ({ratio.mean():.2f}) {np.median(l2):>7.4f} ({l2.mean():.4f})")
        if agent.model.discrete_actions:
            flips = np.array(r["argmax_changed"], dtype=bool)
            line += f"  {flips.sum()}/{n} ({100 * flips.mean():.0f}%)"
        print(line)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, **{
            f"{mod}_{f}": np.array(r[f], dtype=float if f != "argmax_changed" else object)
            for mod, r in rows.items() for f in r
        }, allow_pickle=True)
        print(f"  guardado: {out_path}")


if __name__ == "__main__":
    main()
