"""
Agrega eval_dt.py sobre las 3 semillas del re-entrenamiento con rtg
corregido (METODOLOGIA_DT_HDT.md seccion 2.10, jobs 28426-28443, punto 9
del resumen de proximos pasos).

eval_dt.py ya promedia --num-episodes episodios de UNA corrida; este
script corre eso para cada semilla {1,2,3} de cada (agente, tarea) y
reporta media +/- desviacion ENTRE SEMILLAS del score normalizado D4RL,
que es lo que pide el punto 9 (no solo entre episodios).

Requiere el env `dt-env` (torch + gym + mujoco_py), igual que eval_dt.py.
"""
# eval_dt fija LD_LIBRARY_PATH/MUJOCO_PY_FORCE_CPU y se re-ejecuta con
# os.execve si hace falta -- por eso va primero, antes de numpy/torch/gym.
import eval_dt
from eval_dt import TASKS, rollout, d4rl_normalized_score

import argparse
from pathlib import Path

import gym
import numpy as np
import torch

from agent.dt import DTAgent
from agent.hdt import HDTAgent

SEEDS = [1, 2, 3]
AGENTS = ["dt", "hdt"]
TASK_NAMES = ["halfcheetah", "hopper", "walker2d"]

# Referencia oficial Decision Transformer (Chen et al. 2021, Tabla 2,
# Medium-Expert), ver METODOLOGIA_DT_HDT.md seccion 2.5.
PAPER_REF = {
    "halfcheetah": (86.8, 1.3),
    "hopper": (107.6, 1.8),
    "walker2d": (108.1, 0.2),
}


def snapshot_path(task_name, agent_name, seed, step):
    return (
        Path.home() / "snapshot" / f"{task_name}_medium_expert_{agent_name}_rtgfix"
        / task_name / str(seed) / f"snapshot_{step}.pt"
    )


def eval_one_seed(snapshot, agent_name, task_name, num_episodes, max_steps,
                   episode_length, device, base_seed):
    task = TASKS[task_name]
    target_return = task["eval_targets"][0]

    payload = torch.load(snapshot, map_location=device)
    transformer_cfg = payload["cfg"]
    traj_length = transformer_cfg.traj_length

    agent_cls = DTAgent if agent_name == "dt" else HDTAgent
    agent = agent_cls(
        name=f"{agent_name}_eval",
        obs_shape=(task["obs_dim"],),
        action_shape=(task["action_dim"],),
        device=device,
        lr=1e-4,
        batch_size=1,
        stddev_schedule=0.2,
        use_tb=False,
        transformer_cfg=transformer_cfg,
    )
    # strict=False: ver eval_dt.py (checkpoints previos a la normalizacion
    # de obs no tienen obs_mean/obs_std; no aplica a estos snapshots
    # rtgfix pero se mantiene por consistencia).
    agent.model.load_state_dict(payload["model"], strict=False)
    agent.train(False)

    env = gym.make(task["env_id"])
    try:
        returns = []
        for ep in range(num_episodes):
            _, total_reward, _ = rollout(
                agent, env, task["action_dim"], traj_length, episode_length,
                target_return, max_steps, base_seed + ep, render=False,
            )
            returns.append(total_reward)
    finally:
        env.close()

    returns = np.array(returns)
    norm = d4rl_normalized_score(task_name, returns)
    return norm.mean(), norm.std()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=100000,
                         help="Step del snapshot a evaluar (ultimo de num_grad_steps=100010)")
    parser.add_argument("--num-episodes", type=int, default=10,
                         help="Episodios a promediar por semilla")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--episode-length", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0,
                         help="Semilla base de los episodios de rollout (offset por episodio)")
    args = parser.parse_args()

    print(f"{'tarea':<12} {'agente':<6} {'s1':>8} {'s2':>8} {'s3':>8}   "
          f"{'media +/- std (semillas)':<28} {'paper':<14}")

    all_results = {}
    for task_name in TASK_NAMES:
        for agent_name in AGENTS:
            seed_means = []
            per_seed_cells = []
            for seed in SEEDS:
                path = snapshot_path(task_name, agent_name, seed, args.step)
                if not path.exists():
                    per_seed_cells.append("falta")
                    continue
                mean, _std = eval_one_seed(
                    path, agent_name, task_name, args.num_episodes,
                    args.max_steps, args.episode_length, args.device, args.seed,
                )
                seed_means.append(mean)
                per_seed_cells.append(f"{mean:.2f}")

            if seed_means:
                agg_mean = float(np.mean(seed_means))
                agg_std = float(np.std(seed_means))
                all_results[(task_name, agent_name)] = (agg_mean, agg_std, len(seed_means))
                agg_str = f"{agg_mean:.2f} +/- {agg_std:.2f} (n={len(seed_means)})"
            else:
                agg_str = "sin snapshots"

            pref_mean, pref_std = PAPER_REF[task_name]
            paper_str = f"{pref_mean:.1f} +/- {pref_std:.1f}"
            cells = "  ".join(f"{c:>8}" for c in per_seed_cells)
            print(f"{task_name:<12} {agent_name:<6} {cells}   {agg_str:<28} {paper_str:<14}")

    return all_results


if __name__ == "__main__":
    main()
