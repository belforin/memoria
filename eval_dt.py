"""
Rollout visual: carga un snapshot entrenado de dt/hdt y lo corre en el
entorno gym-mujoco HalfCheetah-v2 en vivo (el mismo entorno subyacente que
D4RL usa para halfcheetah-expert-v2), grabando un video del resultado.

No depende de dmc/dm_control -- usa gym + mujoco_py directamente, igual que
D4RL. Pensado para correr en el env `dt-env` (torch + gym + mujoco_py ya
instalados ahi), no en `maskdp-env` (que no tiene mujoco_py).

Requiere las variables de entorno de mujoco_py; si no estan seteadas, este
script las completa con valores razonables por defecto ANTES de importar
gym (renderizado por CPU, sin necesitar GPU/driver nvidia).
"""
import os
import sys

# mujoco_py necesita LD_LIBRARY_PATH seteado ANTES de que el linker dinámico
# arranque el proceso (mutar os.environ ya con el proceso corriendo no le
# llega al dlopen de sus .so). Si falta, nos reejecutamos con el entorno
# correcto en vez de pedirle al usuario que exporte variables a mano.
_MUJOCO_BIN = os.path.expanduser("~/.mujoco/mujoco210/bin")


def _ensure_mujoco_env():
    env = dict(os.environ)
    changed = False
    if _MUJOCO_BIN not in env.get("LD_LIBRARY_PATH", ""):
        env["LD_LIBRARY_PATH"] = env.get("LD_LIBRARY_PATH", "") + ":" + _MUJOCO_BIN
        changed = True
    if env.get("MUJOCO_PY_FORCE_CPU") != "1":
        env["MUJOCO_PY_FORCE_CPU"] = "1"
        changed = True
    if changed:
        os.execve(sys.executable, [sys.executable] + sys.argv, env)


_ensure_mujoco_env()

import argparse
from collections import deque
from pathlib import Path

import gym
import imageio
import numpy as np
import torch

from agent.dt import DTAgent
from agent.hdt import HDTAgent

# obs_dim/action_dim, id de entorno gym (mismo -v2 que usa D4RL/mujoco_py),
# y targets de retorno de evaluacion / normalizacion D4RL por tarea.
# Targets de retorno: valores oficiales de kzl/decision-transformer
# (gym/experiment.py, env_targets). Random/expert: D4RL (Fu et al. 2020),
# ver METODOLOGIA_DT_HDT.md seccion 2.5.
TASKS = {
    "halfcheetah": dict(
        env_id="HalfCheetah-v2", obs_dim=17, action_dim=6,
        eval_targets=[12000.0, 6000.0],
        d4rl_random=-280.2, d4rl_expert=12135.0,
    ),
    "hopper": dict(
        env_id="Hopper-v2", obs_dim=11, action_dim=3,
        eval_targets=[3600.0, 1800.0],
        d4rl_random=-20.3, d4rl_expert=3234.3,
    ),
    "walker2d": dict(
        env_id="Walker2d-v2", obs_dim=17, action_dim=6,
        eval_targets=[5000.0, 2500.0],
        d4rl_random=1.6, d4rl_expert=4592.3,
    ),
}


def d4rl_normalized_score(task, raw_return):
    t = TASKS[task]
    return 100 * (raw_return - t["d4rl_random"]) / (t["d4rl_expert"] - t["d4rl_random"])


def rollout(agent, env, action_dim, traj_length, episode_length, target_return,
            max_steps, seed, render=False):
    env.seed(seed)
    obs = env.reset()

    obs_hist = deque(maxlen=traj_length)
    action_hist = deque(maxlen=traj_length)
    rtg_hist = deque(maxlen=traj_length)
    timestep_hist = deque(maxlen=traj_length)

    obs_hist.append(obs)
    action_hist.append(np.zeros(action_dim, dtype=np.float32))
    rtg_hist.append(float(target_return))
    timestep_hist.append(0)

    frames = [env.render(mode="rgb_array")] if render else None
    total_reward = 0.0

    for t in range(max_steps):
        obs_arr = np.stack(obs_hist).astype(np.float32)
        action_arr = np.stack(action_hist).astype(np.float32)
        rtg_arr = np.array(rtg_hist, dtype=np.float32).reshape(-1, 1)
        timestep_arr = np.array(timestep_hist, dtype=np.int64)

        with torch.no_grad():
            pred_action = agent.act(obs_arr, action_arr, rtg_arr, timestep_arr, step=t)
        pred_action = np.clip(pred_action, env.action_space.low, env.action_space.high)

        next_obs, reward, done, info = env.step(pred_action)
        total_reward += reward
        if render:
            frames.append(env.render(mode="rgb_array"))

        # la acción predicha reemplaza el placeholder del último paso
        action_hist[-1] = pred_action

        if done:
            break

        next_timestep = min(timestep_hist[-1] + 1, episode_length - 1)
        obs_hist.append(next_obs)
        rtg_hist.append(rtg_hist[-1] - reward)
        action_hist.append(np.zeros(action_dim, dtype=np.float32))
        timestep_hist.append(next_timestep)

    return frames, total_reward, t + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True, help="Path al snapshot_*.pt")
    parser.add_argument("--agent", choices=["dt", "hdt"], required=True)
    parser.add_argument("--task", choices=list(TASKS), default="halfcheetah")
    parser.add_argument("--target-return", type=float, default=None,
                         help="Return-to-go objetivo. Default: el primer target oficial "
                              "de kzl/decision-transformer para --task (ver TASKS).")
    parser.add_argument("--num-episodes", type=int, default=10,
                         help="Episodios a promediar (§2.4 sugiere 10-20 por costo)")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--episode-length", type=int, default=1000,
                         help="Debe coincidir con episode_length usado en entrenamiento (dimensiona el pos_embed)")
    parser.add_argument("--save-video", action="store_true",
                         help="Graba un video solo del primer episodio (no de todos, por costo)")
    parser.add_argument("--out", default="rollout.mp4", help="Path del video de salida")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    task = TASKS[args.task]
    target_return = args.target_return if args.target_return is not None else task["eval_targets"][0]

    payload = torch.load(args.snapshot, map_location=args.device)
    transformer_cfg = payload["cfg"]
    traj_length = transformer_cfg.traj_length

    agent_cls = DTAgent if args.agent == "dt" else HDTAgent
    agent = agent_cls(
        name=f"{args.agent}_eval",
        obs_shape=(task["obs_dim"],),
        action_shape=(task["action_dim"],),
        device=args.device,
        lr=1e-4,
        batch_size=1,
        stddev_schedule=0.2,
        use_tb=False,
        transformer_cfg=transformer_cfg,
    )
    # strict=False: checkpoints previos a la normalizacion de obs
    # (METODOLOGIA_DT_HDT.md seccion 2.8) no tienen los buffers
    # obs_mean/obs_std -- quedan en su default de identidad (mean=0,
    # std=1), que es exactamente como se entrenaron.
    agent.model.load_state_dict(payload["model"], strict=False)
    agent.train(False)

    env = gym.make(task["env_id"])

    returns = []
    for ep in range(args.num_episodes):
        render = args.save_video and ep == 0
        frames, total_reward, n_steps = rollout(
            agent,
            env,
            task["action_dim"],
            traj_length,
            args.episode_length,
            target_return,
            args.max_steps,
            args.seed + ep,
            render=render,
        )
        returns.append(total_reward)
        print(f"  episodio {ep}: pasos={n_steps} retorno={total_reward:.2f}")

        if render:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(str(out_path), frames, fps=args.fps)
            print(f"  video: {out_path}")

    returns = np.array(returns)
    norm_scores = 100 * (returns - task["d4rl_random"]) / (task["d4rl_expert"] - task["d4rl_random"])

    print(
        f"\n{args.task} ({args.agent}, target_return={target_return}): "
        f"retorno {returns.mean():.2f} +/- {returns.std():.2f} | "
        f"score normalizado D4RL {norm_scores.mean():.2f} +/- {norm_scores.std():.2f} "
        f"({args.num_episodes} episodios)"
    )


if __name__ == "__main__":
    main()
