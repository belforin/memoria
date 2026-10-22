"""
Resume los resultados de analysis/lag_ablation.py (metrica 2, METODOLOGIA_DT_HDT.md
seccion 7.16): media ± std entre semillas por (tarea, agente) de los
escalares, y curvas por retraso promediadas sobre semillas.

Uso: python analysis/lag_ablation_summary.py <dir con <agente>_<tarea>_s<seed>.json>
Escribe summary.csv y curves.csv en ese directorio e imprime la tabla.
"""
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

SCALARS = ["d_current_state", "d_all_past_states", "d_all_past_actions", "frac_action_past",
           "mean_lag_past", "lag_90"]
COPY = {"KL": ["pred_eq_prev", "expert_eq_prev", "pred_eq_expert"],
        "RMS": ["rms_pred_prev", "rms_expert_prev", "rms_pred_expert"]}
TASKS = ["cheetah_run", "walker_run", "quadruped_walk", "coinrun"]


def main(d):
    groups = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        m = re.match(r"(.+)_(cheetah_run|walker_run|quadruped_walk|coinrun)_s(\d+)\.json$", os.path.basename(f))
        if m:
            groups[(m.group(2), m.group(1))].append(json.load(open(f)))

    rows, curves = [], []
    for (task, agent) in sorted(groups, key=lambda k: (TASKS.index(k[0]), k[1])):
        runs = groups[(task, agent)]
        metric = runs[0]["metric"]
        row = dict(task=task, agent=agent, n_seeds=len(runs), metric=metric)
        for k in SCALARS + COPY[metric]:
            v = np.array([r[k] for r in runs], float)
            row[k] = f"{v.mean():.4g} ± {v.std():.2g}"
        rows.append(row)
        sc = np.array([r["state_curve"] for r in runs], float)
        ac = np.array([r["action_curve"] for r in runs], float)
        for lag in range(sc.shape[1]):
            curves.append(dict(task=task, agent=agent, lag=lag,
                               state_mean=sc[:, lag].mean(), state_std=sc[:, lag].std(),
                               action_mean=np.nanmean(ac[:, lag]) if lag else "",
                               action_std=np.nanstd(ac[:, lag]) if lag else ""))

    for name, data in [("summary.csv", rows), ("curves.csv", curves)]:
        keys = list(dict.fromkeys(k for r in data for k in r))
        with open(os.path.join(d, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(data)

    for task in TASKS:
        sel = [r for r in rows if r["task"] == task]
        if not sel:
            continue
        cols = ["agent", "n_seeds"] + SCALARS + COPY[sel[0]["metric"]]
        print(f"\n## {task} ({sel[0]['metric']})")
        print("| " + " | ".join(cols) + " |")
        print("|" + "---|" * len(cols))
        for r in sel:
            print("| " + " | ".join(str(r[c]) for c in cols) + " |")
        # primeros retrasos de las curvas (media entre semillas)
        for r in sel:
            c = [x for x in curves if x["task"] == task and x["agent"] == r["agent"]][:6]
            print(f"  {r['agent']:<20} estado k=0..5: " + " ".join(f"{x['state_mean']:.3g}" for x in c)
                  + " | accion k=1..5: " + " ".join(f"{x['action_mean']:.3g}" for x in c[1:]))


if __name__ == "__main__":
    main(sys.argv[1])
