import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.select_qualitative_states import load_cell

CELL_RE = re.compile(r"^traj_(.+)__eval_(.+)\.npz$")
DEFAULT_TOPK_FRACS = (0.05, 0.10, 0.20)


def discover_diagonal_models(sarfa_dir):
    models = {}
    for p in sorted(Path(sarfa_dir).glob("traj_*__eval_*.npz")):
        m = CELL_RE.match(p.name)
        if not m:
            continue
        traj_source, eval_model = m.group(1), m.group(2)
        if traj_source == eval_model:
            models[traj_source] = sarfa_dir
    return models


def topk_mass_fraction(m, frac):
    """
    Fraction of the total (clipped-to-nonnegative, sum-normalized) saliency
    mass contained in the top `frac` fraction of pixels. Complementary to
    entropy: two maps with the same entropy can still differ a lot in how
    much weight concentrates in a small hotspot vs. a spread-out tail --
    this metric targets that directly.
    """
    m = np.clip(m, 0, None)
    total = m.sum()
    if total <= 0:
        return np.nan
    flat = (m / total).flatten()
    n = max(1, round(frac * flat.size))
    top = np.partition(flat, -n)[-n:]
    return float(top.sum())


def main():
    parser = argparse.ArgumentParser(
        description="Concentracion top-k de la saliencia espacial SARFA: fraccion de la masa "
                     "total (answer_maps/dP_maps, normalizados a suma unitaria) que cae en el "
                     "top-k% de pixeles del mapa, por estado/modelo/seed. Complementa la entropia: "
                     "mas sensible a un hotspot agudo en pocos pixeles que a la forma global."
    )
    parser.add_argument("sarfa_dirs", nargs="+",
                         help="Una o mas carpetas con traj_*__eval_*.npz (p.ej. hier + uni juntos)")
    parser.add_argument("--map-key", default="answer_maps", choices=["answer_maps", "dP_maps"])
    parser.add_argument("--topk-fracs", nargs="*", type=float, default=list(DEFAULT_TOPK_FRACS))
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    found = {}
    for d in args.sarfa_dirs:
        for name, src_dir in discover_diagonal_models(d).items():
            if name in found:
                raise SystemExit(f"Modelo '{name}' encontrado en mas de un sarfa_dir "
                                  f"({found[name]} y {src_dir}) -- nombres ambiguos.")
            found[name] = src_dir
    if not found:
        raise SystemExit(f"No se encontraron celdas diagonales en {args.sarfa_dirs}")

    names = args.models if args.models else sorted(found.keys())
    missing = [n for n in names if n not in found]
    if missing:
        raise SystemExit(f"Modelos pedidos pero no encontrados: {missing}\nDisponibles: {sorted(found.keys())}")

    fracs = args.topk_fracs
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.sarfa_dirs[0])
    frac_tag = "-".join(f"{int(round(f * 100))}pct" for f in fracs)
    out_csv = out_dir / f"sarfa_topk_{args.map_key}_{frac_tag}.csv"
    out_npz = out_dir / f"sarfa_topk_{args.map_key}_{frac_tag}_perstate.npz"

    print(f"[sarfa_topk] sarfa_dirs={args.sarfa_dirs} map_key={args.map_key} fracs={fracs}")
    print(f"[sarfa_topk] modelos: {names}\n")

    rows = []
    perstate = {}
    perstate_meta = {}
    for name in names:
        cell = load_cell(found[name], name, name)
        maps = cell[args.map_key]  # (N, H/d, W/d)
        arch = str(cell["arch"])
        seed = int(cell["seed"])

        per_frac_values = {f: np.array([topk_mass_fraction(m, f) for m in maps]) for f in fracs}
        n = int(maps.shape[0])

        row = {"model": name, "arch": arch, "seed": seed, "map_key": args.map_key, "n_states": n}
        print(f"  {name} (arch={arch}, seed={seed}): n_states={n}")
        for f in fracs:
            vals = per_frac_values[f]
            mean_v, std_v = float(np.nanmean(vals)), float(np.nanstd(vals))
            row[f"mean_top{int(round(f * 100))}pct"] = mean_v
            row[f"std_top{int(round(f * 100))}pct"] = std_v
            print(f"      top{int(round(f * 100))}%: mean={mean_v:.4f} std={std_v:.4f}")
            perstate[f"{name}__top{int(round(f * 100))}pct"] = vals
        rows.append(row)
        perstate_meta[name] = f"{arch}|{seed}"

    fieldnames = ["model", "arch", "seed", "map_key", "n_states"]
    for f in fracs:
        fieldnames += [f"mean_top{int(round(f * 100))}pct", f"std_top{int(round(f * 100))}pct"]
    with open(out_csv, "w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[sarfa_topk] -> {out_csv}")

    np.savez_compressed(
        out_npz,
        model_names=np.array(names),
        model_meta=np.array([perstate_meta[n] for n in names]),
        topk_fracs=np.array(fracs),
        **perstate,
    )
    print(f"[sarfa_topk] -> {out_npz} (por estado, para el violin)")


if __name__ == "__main__":
    main()
