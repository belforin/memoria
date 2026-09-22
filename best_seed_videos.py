"""
Para cada (agente, tarea) con las 3 semillas del re-entrenamiento rtgfix
disponibles (METODOLOGIA_DT_HDT.md seccion 2.10, jobs 28426-28443/28687),
evalua las semillas {1,2,3}, elige la de mejor score D4RL normalizado
(promedio sobre --num-episodes) y graba en video el primer episodio de esa
semilla ganadora. Si alguna semilla no tiene snapshot, la salta.

Requiere el env `dt-env`, igual que eval_dt.py/eval_seeds.py.
"""
import eval_dt
from eval_dt import TASKS, rollout, d4rl_normalized_score

import argparse
from pathlib import Path

import gym
import imageio
import numpy as np
import torch

from agent.dt import DTAgent
from agent.hdt import HDTAgent

SEEDS = [1, 2, 3]
COMBOS = [
    ("halfcheetah", "dt"), ("halfcheetah", "hdt"),
    ("hopper", "dt"), ("hopper", "hdt"),
    ("walker2d", "dt"), ("walker2d", "hdt"),
]


def snapshot_path(task_name, agent_name, seed, step):
    return (
        Path.home() / "snapshot" / f"{task_name}_medium_expert_{agent_name}_rtgfix"
        / task_name / str(seed) / f"snapshot_{step}.pt"
    )


def eval_seed(snapshot, agent_name, task_name, num_episodes, max_steps,
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
    agent.model.load_state_dict(payload["model"], strict=False)
    agent.train(False)

    env = gym.make(task["env_id"])
    try:
        returns = []
        ep0_frames = None
        for ep in range(num_episodes):
            render = ep == 0
            frames, total_reward, _ = rollout(
                agent, env, task["action_dim"], traj_length, episode_length,
                target_return, max_steps, base_seed + ep, render=render,
            )
            returns.append(total_reward)
            if render:
                ep0_frames = frames
    finally:
        env.close()

    returns = np.array(returns)
    norm = d4rl_normalized_score(task_name, returns)
    return norm.mean(), norm.std(), ep0_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=100000)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--episode-length", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--out-dir", default="eval_results/videos_best_seed_rtgfix")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'tarea':<12} {'agente':<6} {'s1':>8} {'s2':>8} {'s3':>8}   ganadora")

    for task_name, agent_name in COMBOS:
        per_seed = {}
        cells = []
        for seed in SEEDS:
            path = snapshot_path(task_name, agent_name, seed, args.step)
            if not path.exists():
                cells.append("falta")
                continue
            mean, _std, frames = eval_seed(
                path, agent_name, task_name, args.num_episodes,
                args.max_steps, args.episode_length, args.device, args.seed,
            )
            per_seed[seed] = (mean, frames)
            cells.append(f"{mean:.2f}")

        if not per_seed:
            print(f"{task_name:<12} {agent_name:<6} " + "  ".join(f"{c:>8}" for c in cells) + "   sin snapshots")
            continue

        best_seed = max(per_seed, key=lambda s: per_seed[s][0])
        best_mean, best_frames = per_seed[best_seed]
        n_available = len(per_seed)
        note = "" if n_available == len(SEEDS) else f" (solo {n_available}/{len(SEEDS)} semillas disponibles)"

        video_path = out_dir / f"{task_name}_{agent_name}_seed{best_seed}.mp4"
        imageio.mimsave(str(video_path), best_frames, fps=args.fps)

        print(
            f"{task_name:<12} {agent_name:<6} " + "  ".join(f"{c:>8}" for c in cells)
            + f"   seed{best_seed} ({best_mean:.2f}){note} -> {video_path}"
        )


if __name__ == "__main__":
    main()
