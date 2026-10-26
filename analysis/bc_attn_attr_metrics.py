"""
Metricas de AttAttr de la tesis de Benjamin (seccion 4.3.1, Tablas 4.9, 4.10 y
C.1) sobre las corridas de analysis/bc_attn_attr.py. Reusa
analysis/receptive_field.py (copiado tal cual de MaskDP_public) y replica la
agregacion de sus scripts:
  - campo receptivo por estado x capa x cabeza: receptive_field_diagnostics
    (normaliza por el maximo del bloque, descarta atribucion negativa,
    exclude_self_loops=True como en su decoder) -> mean_distance;
  - por semilla: mediana sobre estados por (capa, cabeza)
    (receptive_field_aggregate.py), media sobre cabezas por capa
    (receptive_field_scalar.py);
  - peso de masa por capa: sum|attr| de la capa / total, por estado, promedio
    sobre estados por semilla (decoder_layer_weight.py);
  - campo receptivo ponderado por semilla = sum_capa peso * RF_capa
    (receptive_field_weighted_scalar.py);
  - Herfindahl: M = sum_{capa, cabeza} |attr|, p = M / sum(M), H = sum p^2, por
    estado; media sobre estados por semilla (concentration_batch.py);
  - entre semillas: media +- desviacion muestral (ddof=1).

Conjuntos de capas que se reportan (sus modelos tienen un decodificador
identico; los nuestros no, ver analysis/bc_attn_attr.py):
  - "last_joint": la ultima capa del stack conjunto, la mas cercana al logit en
    ambas topologias (mismo tamano de matriz): la comparacion directa;
  - "joint": todas las capas del stack conjunto (BC-uni 5, BC-hier 1);
  - "all": todas las capas de atencion del modelo (BC-hier suma los stacks por
    modalidad), con pesos de masa sobre el total.

Uso:
    python analysis/bc_attn_attr_metrics.py --out-dir <dir> \
        RUN_DIR_1 RUN_DIR_2 ...   (cada uno con manifest.json + index.csv + raw/)
"""
import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.receptive_field import receptive_field_diagnostics  # noqa: E402


def stream_timesteps(stack, n_s):
    """Paso relativo de cada posicion: stack conjunto intercalado (pos // 2) o stream por modalidad"""
    if stack == "joint":
        return np.arange(2 * n_s) // 2
    return np.arange(n_s)


def rf_or_nan(mat, ts):
    try:
        return receptive_field_diagnostics(mat, ts, ts, exclude_self_loops=True)["mean_distance"]
    except AssertionError:  # sin atribucion positiva fuera de la diagonal (p. ej. n_s = 1)
        return float("nan")


def herfindahl(stacked):
    """stacked: [n_layer, n_head, L, L] -> H de la masa |attr| sumada sobre capas y cabezas"""
    M = np.abs(stacked).sum(axis=(0, 1))
    total = M.sum()
    if total <= 0:
        return float("nan")
    p = M / total
    return float((p ** 2).sum())


def process_run(run_dir):
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    stacks = manifest["attn_stacks"]                # {stack: n_layer}, orden de entrada a salida
    layers = [(s, l) for s, n in stacks.items() for l in range(n)]
    last_joint = ("joint", stacks["joint"] - 1)

    rf = defaultdict(list)          # (stack, layer, head) -> [rf por estado]
    mass = []                       # por estado: {(stack, layer): sum|attr|}
    H_joint, H_last = [], []
    n_states = 0
    with open(run_dir / "index.csv", newline="") as f:
        for row in csv.DictReader(f):
            if row["error"]:
                continue
            d = np.load(row["npz_path"])
            n_s = int(d["n_s"])
            m_state = {}
            for stack in stacks:
                attr = d[f"attr_{stack}"]               # [n_layer, n_head, L, L]
                ts = stream_timesteps(stack, n_s)
                for layer in range(attr.shape[0]):
                    m_state[(stack, layer)] = float(np.abs(attr[layer]).sum())
                    for head in range(attr.shape[1]):
                        rf[(stack, layer, head)].append(rf_or_nan(attr[layer, head], ts))
            mass.append(m_state)
            H_joint.append(herfindahl(d["attr_joint"]))
            H_last.append(herfindahl(d["attr_joint"][-1:]))
            n_states += 1

    n_head = manifest["n_head"]
    rf_layer = {
        (s, l): float(np.mean([np.nanmedian(rf[(s, l, h)]) for h in range(n_head)])) for (s, l) in layers
    }
    rf_head = {k: float(np.nanmedian(v)) for k, v in rf.items()}

    def weights(subset):
        fr = defaultdict(list)
        for m_state in mass:
            total = sum(m_state[k] for k in subset)
            for k in subset:
                fr[k].append(m_state[k] / total if total > 0 else np.nan)
        return {k: float(np.nanmean(v)) for k, v in fr.items()}

    joint_layers = [k for k in layers if k[0] == "joint"]
    w_joint, w_all = weights(joint_layers), weights(layers)
    out = dict(
        model_name=manifest["model_name"], topology=manifest["topology"], n_states=n_states,
        rf_last_joint=rf_layer[last_joint],
        rf_joint_simple=float(np.mean([rf_layer[k] for k in joint_layers])),
        rf_joint_weighted=float(sum(w_joint[k] * rf_layer[k] for k in joint_layers)),
        rf_all_simple=float(np.mean([rf_layer[k] for k in layers])),
        rf_all_weighted=float(sum(w_all[k] * rf_layer[k] for k in layers)),
        H_joint=float(np.nanmean(H_joint)), H_last_joint=float(np.nanmean(H_last)),
        rf_layer={f"{s}{l}": v for (s, l), v in rf_layer.items()},
        rf_head={f"{s}{l}h{h}": v for (s, l, h), v in rf_head.items()},
        w_joint={f"{s}{l}": v for (s, l), v in w_joint.items()},
        w_all={f"{s}{l}": v for (s, l), v in w_all.items()},
    )
    return out


SCALARS = ["rf_last_joint", "rf_joint_simple", "rf_joint_weighted", "rf_all_simple", "rf_all_weighted",
           "H_joint", "H_last_joint"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_seed = [process_run(r) for r in args.run_dirs]
    (out_dir / "attattr_per_seed.json").write_text(json.dumps(per_seed, indent=2))

    with open(out_dir / "attattr_per_seed.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model_name", "topology", "n_states"] + SCALARS)
        for r in per_seed:
            w.writerow([r["model_name"], r["topology"], r["n_states"]] + [f"{r[k]:.6g}" for k in SCALARS])

    # Agrupa por modelo (nombre sin _s<semilla>), no por topologia: bc_hier,
    # bc_hier223 y bc_hier223_enc128 son todos "hier" pero con capas distintas.
    by_topo = defaultdict(list)
    for r in per_seed:
        by_topo[re.sub(r"_s\d+$", "", r["model_name"])].append(r)
    lines = []
    for topo, rs in sorted(by_topo.items()):
        lines.append(f"== {topo} ({len(rs)} semillas: {', '.join(r['model_name'] for r in rs)})")
        lines.append(f"   estados por semilla: {[r['n_states'] for r in rs]}")
        for k in SCALARS:
            vals = np.array([r[k] for r in rs])
            std = vals.std(ddof=1) if len(vals) > 1 else float("nan")
            lines.append(f"   {k:18s} {vals.mean():.4f} +- {std:.4f}   crudos {np.round(vals, 4).tolist()}")
        for key in rs[0]["rf_layer"]:
            rfs = np.array([r["rf_layer"][key] for r in rs])
            wj = np.array([r["w_joint"].get(key, np.nan) for r in rs])
            wa = np.array([r["w_all"][key] for r in rs])
            lines.append(f"   capa {key:8s} RF {rfs.mean():.3f} +- {rfs.std(ddof=1) if len(rs) > 1 else 0:.3f} | "
                         f"peso joint {np.nanmean(wj):.3f} | peso all {wa.mean():.3f}")
    text = "\n".join(lines)
    print(text)
    (out_dir / "attattr_summary.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
