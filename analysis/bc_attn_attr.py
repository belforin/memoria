"""
AttAttr (Hao et al. 2021) sobre los modelos BC autoregresivos (agent/bc_ar.py),
adaptado del pipeline de la tesis de Benjamin (MaskDP_public:
analysis/attn_attr.py, run_attn_attr.py, run_attn_attr_batch.py). Misma
matematica (gradiente integrado sobre los pesos de atencion, m pasos de
Riemann, objetivo = logit de la accion argmax del modelo), mismo esquema de
salida (un .npz por estado + index.csv + manifest.json) y mismas grabaciones
(states_*.pkl de analysis/eval_bct_levels.py), asi eval_sarfa.py se une 1:1
contra el index.csv de aca.

Diferencia de arquitectura (belforin/memoria METODOLOGIA_DT_HDT.md seccion 7):
Benjamin atribuye sobre el decodificador, identico en sus dos topologias. Los
modelos BC no tienen decodificador: la accion a_t se lee del token s_t a la
salida del stack conjunto causal. Se atribuye sobre TODAS las capas de
atencion del modelo:
  - "joint": stack conjunto sobre [s_0, a_0, s_1, a_1, ...] (BC-uni: todas sus
    capas; BC-hier: la capa de fusion). Matriz [2 n_s, 2 n_s]; el ultimo token
    (a_{n_s-1}) es el relleno que no se usa para predecir.
  - "obs"/"act" (solo BC-hier): stacks por modalidad, [n_s, n_s].
El token objetivo es s_{n_s-1} (posicion 2(n_s-1) del stack conjunto), el
analogo del a_0* enmascarado de Benjamin.

Las m interpolaciones van en un solo batch (en vez del bucle de Benjamin): el
modelo no mezcla filas del batch (LayerNorm por token, dropout apagado), asi
que los gradientes son los mismos (verificado en test_bc_ar.py).

Uso:
    python analysis/bc_attn_attr.py --path <snapshot.pt> --model-name bc_uni_1 \
        --states-dir <record-dir>/test --out-dir <dir> [--m 20] [--device cuda] [--limit N]
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.bc_ar import BCAREvalAgent  # noqa: E402
from analysis.trajectory_recorder import load_buffers, load_trajectory_states  # noqa: E402

OBS_SHAPE = (64, 64, 3)
ACTION_SHAPE = (1,)

INDEX_FIELDS = [
    "name", "level", "state_index", "t", "action_idx", "target_mode", "golden_action", "agreed",
    "p_max", "p_golden", "entropy", "logit_range", "n_s", "n_a", "K", "topology", "n_head",
    "num_actions", "m", "npz_path", "error",
]


def attn_attr_layer(agent, stack, tar_layer, action_idx, m=20):
    """
    Atribucion [n_head, L, L] de una capa respecto del logit de action_idx en la
    posicion de la accion actual. Igual que attn_attr_layer de Benjamin:
    Attr(A) = A * (1/m) sum_k dF(k/m A)/dA.
    """
    n_s = len(agent._obs_buffer)
    with torch.no_grad():
        _, att = agent._forward_for_attr(stack, tar_layer, capture_att=True)
    att = att.detach()                                                   # (1, H, L, L)
    alphas = torch.arange(1, m + 1, device=att.device, dtype=att.dtype) / m
    tmp_att = (alphas[:, None, None, None] * att).clone().requires_grad_(True)  # (m, H, L, L)
    pred_a, _ = agent._forward_for_attr(stack, tar_layer, tmp_att=tmp_att)
    pred_a[:, n_s - 1, action_idx].sum().backward()
    return (att[0] * tmp_att.grad.mean(dim=0)).detach()


def attn_attr_all(agent, action_idx, m=20):
    """{stack: [n_layer, n_head, L, L]} para todos los stacks del modelo"""
    return {
        stack: torch.stack([attn_attr_layer(agent, stack, layer, action_idx, m) for layer in range(n_layer)])
        for stack, n_layer in agent.mdp.attn_stacks().items()
    }


def process_state(agent, state, m=20):
    load_buffers(agent, state["obs_buffer"], state["action_buffer"])
    n_s, n_a = len(agent._obs_buffer), len(agent._action_buffer)

    logits = agent.logits_for()
    p = torch.softmax(logits, dim=-1)
    logp = torch.log_softmax(logits, dim=-1)
    action_idx = int(logits.argmax().item())
    golden = int(state["traj_action"]) if state.get("traj_action") is not None else -1

    attr = attn_attr_all(agent, action_idx, m)
    with torch.no_grad():
        n_joint = agent.mdp.attn_stacks()["joint"]
        _, raw_att = agent._forward_for_attr("joint", n_joint - 1, capture_att=True)

    meta = dict(
        t=int(state.get("t", -1)), action_idx=action_idx, target_mode="argmax", golden_action=golden,
        agreed=bool(golden == action_idx), p_max=float(p.max()),
        p_golden=float(p[golden]) if golden >= 0 else float("nan"),
        entropy=float(-(p * logp).sum()), logit_range=float(logits.max() - logits.min()),
        n_s=n_s, n_a=n_a, K=agent.K, topology=agent.mdp.topology, n_head=int(agent.config.n_head),
        num_actions=agent.num_actions, m=m,
    )
    arrays = {f"attr_{k}": v.cpu().numpy().astype(np.float32) for k, v in attr.items()}
    arrays["raw_att_joint_last"] = raw_att[0].cpu().numpy().astype(np.float32)
    return meta, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--path", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--states-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--m", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None, help="procesa solo N estados (piloto)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    agent = BCAREvalAgent(obs_shape=OBS_SHAPE, action_shape=ACTION_SHAPE, device=torch.device(args.device),
                          temperature=1.0, sample=False, path=args.path)
    agent.eval()
    stacks = agent.mdp.attn_stacks()

    names = sorted(p.name[len("states_"):-len(".pkl")] for p in Path(args.states_dir).glob("states_*.pkl"))
    assert names, f"no hay states_*.pkl en {args.states_dir}"

    manifest = dict(
        status="running", model_name=args.model_name, snapshot_path=args.path, states_dir=args.states_dir,
        topology=agent.mdp.topology, attn_stacks=stacks, n_head=int(agent.config.n_head), K=agent.K,
        m=args.m, target="argmax", n_pkls=len(names),
    )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    n_done = n_err = 0
    t0 = time.time()
    with open(out_dir / "index.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        for name in names:
            for level, states in load_trajectory_states(args.states_dir, name).items():
                for idx, state in enumerate(states):
                    if args.limit is not None and n_done + n_err >= args.limit:
                        break
                    npz_path = raw_dir / f"{name}__l{level}__s{idx:04d}.npz"
                    row = dict(name=name, level=level, state_index=idx, npz_path=str(npz_path), error="")
                    try:
                        meta, arrays = process_state(agent, state, args.m)
                        np.savez_compressed(npz_path, **arrays, **meta, level=level, state_index=idx,
                                            model_name=args.model_name)
                        row.update({k: v for k, v in meta.items() if k in INDEX_FIELDS})
                        n_done += 1
                    except Exception as e:  # registra y sigue, como run_attn_attr_batch.py
                        row["error"] = f"{type(e).__name__}: {e}"
                        n_err += 1
                    writer.writerow(row)
                    f.flush()
                    if (n_done + n_err) % 200 == 0:
                        dt = time.time() - t0
                        print(f"[bc_attn_attr] {n_done + n_err} estados ({n_err} errores) "
                              f"en {dt:.0f}s ({dt / (n_done + n_err):.3f}s/estado)", flush=True)

    dt = time.time() - t0
    manifest.update(status="done", n_ok=n_done, n_error=n_err, seconds=round(dt, 1))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[bc_attn_attr] {args.model_name}: {n_done} ok, {n_err} errores, {dt:.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()
