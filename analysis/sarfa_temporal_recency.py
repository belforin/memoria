import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.select_qualitative_states import load_cell

CELL_RE = re.compile(r"^traj_(.+)__eval_(.+)\.npz$")


def discover_diagonal_models(sarfa_dir):
    """Finds {name: sarfa_dir} for diagonal cells (traj_X__eval_X.npz) in sarfa_dir."""
    models = {}
    for p in sorted(Path(sarfa_dir).glob("traj_*__eval_*.npz")):
        m = CELL_RE.match(p.name)
        if not m:
            continue
        traj_source, eval_model = m.group(1), m.group(2)
        if traj_source == eval_model:
            models[traj_source] = sarfa_dir
    return models


def temporal_shares(temporal_arr, n_s_arr, k_max):
    """
    For each state, takes the valid (non-NaN) prefix of length n_s, clips to
    non-negative (SARFA `answer`/`dP` can be negative; only the positive part
    is treated as saliency mass, same convention as map_entropy), normalizes
    to sum 1, then reverses it so index 0 = current frame (lag 0), index 1 =
    one step back (lag 1), etc. Returns a (N, k_max) matrix padded with NaN
    past each state's own n_s (fewer past frames were available).
    """
    n = temporal_arr.shape[0]
    out = np.full((n, k_max), np.nan, dtype=np.float64)
    for i in range(n):
        n_s = int(n_s_arr[i])
        valid = np.clip(temporal_arr[i, :n_s], 0, None)
        total = valid.sum()
        if total <= 0:
            continue
        share = (valid / total)[::-1]  # lag 0 = current frame (last valid index)
        out[i, :n_s] = share
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Recencia temporal de SARFA: como se reparte la masa de saliencia "
                     "temporal (temporal_answer/temporal_dP) entre el frame actual (lag=0) "
                     "y los frames pasados del buffer (lag=1,2,...), por modelo/seed. "
                     "Barato: reusa temporal_answer/temporal_dP y n_s ya guardados por eval_sarfa.py."
    )
    parser.add_argument("sarfa_dirs", nargs="+",
                         help="Una o mas carpetas con traj_*__eval_*.npz (p.ej. sarfa_coinrun_official "
                              "sarfa_coinrun_diagonal para combinar hier y uni en un solo resultado)")
    parser.add_argument("--map-key", default="temporal_answer", choices=["temporal_answer", "temporal_dP"])
    parser.add_argument("--models", nargs="*", default=None,
                         help="Nombres de modelo a incluir. Default: todas las celdas diagonales "
                              "encontradas en los sarfa_dirs dados.")
    parser.add_argument("--out-dir", default=None,
                         help="Carpeta de salida para CSV/npz (default: la primera de sarfa_dirs)")
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

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.sarfa_dirs[0])
    out_csv = out_dir / f"sarfa_temporal_recency_{args.map_key}.csv"
    out_npz = out_dir / f"sarfa_temporal_recency_{args.map_key}_curves.npz"

    print(f"[sarfa_temporal_recency] sarfa_dirs={args.sarfa_dirs} map_key={args.map_key}")
    print(f"[sarfa_temporal_recency] modelos: {names}\n")

    rows = []
    curves = {}
    curve_meta = {}
    k_max = None
    for name in names:
        cell = load_cell(found[name], name, name)
        temporal_arr = cell[args.map_key]  # (N, K_max)
        n_s_arr = cell["n_s"]  # (N,)
        if k_max is None:
            k_max = temporal_arr.shape[1]
        assert temporal_arr.shape[1] == k_max, f"K_max mismatch for {name}: {temporal_arr.shape[1]} vs {k_max}"

        shares = temporal_shares(temporal_arr, n_s_arr, k_max)  # (N, k_max), NaN-padded
        lag_mean = np.nanmean(shares, axis=0)
        lag_std = np.nanstd(shares, axis=0)
        lag_count = np.sum(~np.isnan(shares), axis=0)

        current_frame_share = shares[:, 0]
        current_frame_share = current_frame_share[~np.isnan(current_frame_share)]
        cf_mean = float(np.mean(current_frame_share))
        cf_std = float(np.std(current_frame_share))
        n = int(current_frame_share.size)
        arch = str(cell["arch"])
        seed = int(cell["seed"])

        print(f"  {name} (arch={arch}, seed={seed}): n_states={n} "
              f"current_frame_share mean={cf_mean:.4f} std={cf_std:.4f}")
        for lag in range(min(4, k_max)):
            print(f"      lag={lag}: mean_share={lag_mean[lag]:.4f} n={int(lag_count[lag])}")

        rows.append({
            "model": name, "arch": arch, "seed": seed, "map_key": args.map_key,
            "n_states": n, "current_frame_share_mean": cf_mean, "current_frame_share_std": cf_std,
        })
        curves[name] = (lag_mean, lag_std, lag_count)
        curve_meta[name] = f"{arch}|{seed}"

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "arch", "seed", "map_key", "n_states",
                                                "current_frame_share_mean", "current_frame_share_std"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[sarfa_temporal_recency] -> {out_csv}")

    npz_payload = {
        "model_names": np.array(list(curves.keys())),
        "model_meta": np.array([curve_meta[n] for n in curves]),
        "lag": np.arange(k_max),
    }
    for n, (mean_c, std_c, count_c) in curves.items():
        npz_payload[f"lag_mean__{n}"] = mean_c
        npz_payload[f"lag_std__{n}"] = std_c
        npz_payload[f"lag_count__{n}"] = count_c
    np.savez_compressed(out_npz, **npz_payload)
    print(f"[sarfa_temporal_recency] -> {out_npz} (curvas por lag, para el render)")


if __name__ == "__main__":
    main()
