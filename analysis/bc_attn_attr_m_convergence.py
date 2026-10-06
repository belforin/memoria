"""
Convergencia de AttAttr respecto del numero de pasos m, como la Tabla 3.10 de
la tesis de Benjamin: para cada m, Spearman entre el ranking de conexiones y el
de la referencia m=50, y coincidencia entre las 10 conexiones de mayor
atribucion. Se mide sobre la atribucion de cada capa (sumada sobre cabezas),
promediando sobre capas y estados. Benjamin lo reporta en el cuantil de menor
confianza (p_max < 0.2); aca se reporta para todos los estados y para ese
cuantil, si hay estados en el.

Uso:
    python analysis/bc_attn_attr_m_convergence.py --path <snapshot.pt> \
        --states-dir <record-dir>/test [--n-states 60] [--device cuda]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.bc_ar import BCAREvalAgent  # noqa: E402
from analysis.bc_attn_attr import attn_attr_layer  # noqa: E402
from analysis.trajectory_recorder import load_buffers, load_trajectory_states, select_quantile_states  # noqa: E402

M_VALUES = (5, 10, 20)
M_REF = 50


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--path", required=True)
    parser.add_argument("--states-dir", required=True)
    parser.add_argument("--n-states", type=int, default=60)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    agent = BCAREvalAgent(obs_shape=(64, 64, 3), action_shape=(1,), device=torch.device(args.device),
                          temperature=1.0, sample=False, path=args.path)
    agent.eval()

    states = []
    for p in sorted(Path(args.states_dir).glob("states_*.pkl")):
        for steps in load_trajectory_states(args.states_dir, p.name[len("states_"):-len(".pkl")]).values():
            states.extend(steps)
    states = [s for s in states if len(s["obs_buffer"]) >= 2]  # con n_s = 1 no hay ranking que comparar
    states = select_quantile_states(states, args.n_states)

    rows = []  # (p_max, m, spearman, top10)
    for state in states:
        load_buffers(agent, state["obs_buffer"], state["action_buffer"])
        logits = agent.logits_for()
        a_idx = int(logits.argmax())
        p_max = float(torch.softmax(logits, -1).max())
        for stack, n_layer in agent.mdp.attn_stacks().items():
            for layer in range(n_layer):
                ref = attn_attr_layer(agent, stack, layer, a_idx, m=M_REF).sum(0).flatten().cpu().numpy()
                nz = np.flatnonzero(ref != 0)  # celdas causales con atribucion (las demas son 0 en todo m)
                if len(nz) < 10:
                    continue
                top_ref = set(nz[np.argsort(-ref[nz])[:10]])
                for m in M_VALUES:
                    a = attn_attr_layer(agent, stack, layer, a_idx, m=m).sum(0).flatten().cpu().numpy()
                    rho = spearmanr(a[nz], ref[nz]).correlation
                    top = set(nz[np.argsort(-a[nz])[:10]])
                    rows.append((p_max, m, rho, len(top & top_ref)))

    rows = np.array(rows)
    print(f"[m_convergence] {agent.mdp.topology}: {len(states)} estados, referencia m={M_REF}")
    for label, mask in [("todos", np.ones(len(rows), bool)), ("p_max < 0.2", rows[:, 0] < 0.2)]:
        for m in M_VALUES:
            sel = mask & (rows[:, 1] == m)
            if sel.sum() == 0:
                print(f"  {label:12s} m={m:3d}: sin estados")
                continue
            print(f"  {label:12s} m={m:3d}: Spearman {np.nanmean(rows[sel, 2]):.4f} | "
                  f"top-10 {rows[sel, 3].mean():.2f}/10 | n={int(sel.sum())} (capa x estado)")


if __name__ == "__main__":
    main()
