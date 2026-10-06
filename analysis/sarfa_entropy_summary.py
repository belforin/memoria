import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.compare_saliency import map_entropy
from analysis.select_qualitative_states import load_cell

CELL_RE = re.compile(r"^traj_(.+)__eval_(.+)\.npz$")


def discover_diagonal_models(sarfa_dir):
    """Finds {name: npz_path} for diagonal cells (traj_X__eval_X.npz) in sarfa_dir."""
    models = {}
    for p in sorted(Path(sarfa_dir).glob("traj_*__eval_*.npz")):
        m = CELL_RE.match(p.name)
        if not m:
            continue
        traj_source, eval_model = m.group(1), m.group(2)
        if traj_source == eval_model:
            models[traj_source] = p
    return models


def entropy_per_state(cell, map_key):
    return np.array([map_entropy(m) for m in cell[map_key]], dtype=np.float64)


def entropy_by_level(cell, entropies):
    per_level = {}
    for level in np.unique(cell["levels"]):
        mask = cell["levels"] == level
        per_level[int(level)] = float(np.mean(entropies[mask]))
    return per_level


def main():
    parser = argparse.ArgumentParser(
        description="Entropia de Shannon de los mapas de saliencia SARFA (answer_maps/dP_maps), "
                     "normalizados a suma unitaria, por modelo/seed. Lee las celdas diagonales "
                     "traj_X__eval_X.npz escritas por eval_sarfa.py phase=evaluate."
    )
    parser.add_argument("sarfa_dir", help="Carpeta con los traj_*__eval_*.npz (p.ej. sarfa_coinrun_official)")
    parser.add_argument("--map-key", default="answer_maps", choices=["answer_maps", "dP_maps"])
    parser.add_argument("--models", nargs="*", default=None,
                         help="Nombres de modelo a incluir (p.ej. hier_seed1 hier_seed2 hier_seed3). "
                              "Default: todas las celdas diagonales encontradas en sarfa_dir.")
    parser.add_argument("--out-csv", default=None,
                         help="Ruta del CSV de salida (default: <sarfa_dir>/sarfa_entropy_<map_key>.csv)")
    args = parser.parse_args()

    found = discover_diagonal_models(args.sarfa_dir)
    if not found:
        raise SystemExit(f"No se encontraron celdas diagonales traj_X__eval_X.npz en {args.sarfa_dir}")

    names = args.models if args.models else sorted(found.keys())
    missing = [n for n in names if n not in found]
    if missing:
        raise SystemExit(f"Modelos pedidos pero no encontrados en {args.sarfa_dir}: {missing}\n"
                          f"Disponibles: {sorted(found.keys())}")

    out_csv = Path(args.out_csv) if args.out_csv else Path(args.sarfa_dir) / f"sarfa_entropy_{args.map_key}.csv"
    out_perstate = Path(args.sarfa_dir) / f"sarfa_entropy_{args.map_key}_perstate.npz"

    rows = []
    perstate = {}
    perstate_meta = {}
    print(f"[sarfa_entropy] sarfa_dir={args.sarfa_dir} map_key={args.map_key}")
    print(f"[sarfa_entropy] modelos: {names}\n")

    all_means = []
    for name in names:
        cell = load_cell(args.sarfa_dir, name, name)
        entropies = entropy_per_state(cell, args.map_key)
        mean_e = float(np.mean(entropies))
        std_e = float(np.std(entropies))
        n = int(entropies.size)
        arch = str(cell["arch"])
        seed = int(cell["seed"])
        all_means.append(mean_e)
        perstate[name] = entropies
        perstate_meta[name] = f"{arch}|{seed}"

        per_level = entropy_by_level(cell, entropies)
        print(f"  {name} (arch={arch}, seed={seed}): n_states={n} "
              f"mean_entropy={mean_e:.4f} std={std_e:.4f}")
        for level, lv_mean in sorted(per_level.items()):
            print(f"      level={level}: mean_entropy={lv_mean:.4f}")

        rows.append({
            "model": name, "arch": arch, "seed": seed, "map_key": args.map_key,
            "n_states": n, "mean_entropy": mean_e, "std_entropy": std_e,
        })

    print(f"\n[sarfa_entropy] promedio entre seeds (n={len(names)}): "
          f"mean={np.mean(all_means):.4f} std={np.std(all_means):.4f}")

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "arch", "seed", "map_key", "n_states",
                                                "mean_entropy", "std_entropy"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[sarfa_entropy] -> {out_csv}")

    np.savez_compressed(
        out_perstate,
        model_names=np.array(list(perstate.keys())),
        model_meta=np.array([perstate_meta[n] for n in perstate]),  # "arch|seed", aligned with model_names
        **{f"entropy__{n}": v for n, v in perstate.items()},
    )
    print(f"[sarfa_entropy] -> {out_perstate} (entropia por estado, para el violin plot)")


if __name__ == "__main__":
    main()
