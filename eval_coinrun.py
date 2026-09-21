"""
Rollout closed-loop de un snapshot dt_coinrun/hdt_coinrun contra el entorno
real de Procgen (CoinRun), corriendo cada paso via DTAgent.act()/
HDTAgent.act() -- return-conditioned, igual mecanismo que eval_dt.py para
D4RL, pero con acciones discretas y obs de pixeles.

Puerto de eval_bct.py/agent/mdp_bct.py de Benjamin Mancilla (rama
upstream/hier-procgen) a nuestros DTAgent/HDTAgent -- reusa su
VecExtractDictObs, el diccionario PROCGEN de normalizacion y los splits
train/val/test del protocolo de Mediratta et al. (ICLR 2024). Ver
METODOLOGIA_DT_HDT.md, seccion 3.

Corre en el conda env `procgen-env` (clon de
/home/bmancilla/miniconda3/envs/maskdp_procgen, ya verificado funcionando
headless en este cluster), no en maskdp-env/dt-env.
"""
import argparse
from collections import deque
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import procgen
import torch

from agent.dt import DTAgent
from agent.hdt import HDTAgent


class VecExtractDictObs:
    """
    Copia sin dependencias extra (gym mas nuevo) de:
    `baselines.common.vec_env.VecExtractDictObs` (openai/baselines).
    Identica a la de eval_bct.py (rama upstream/hier-procgen).
    """

    def __init__(self, venv, key):
        self.venv = venv
        self.key = key
        self.observation_space = venv.observation_space.spaces[key]
        self.action_space = venv.action_space
        self.num_envs = venv.num_envs

    def reset(self):
        obs = self.venv.reset()
        return obs[self.key]

    def step(self, action):
        obs, rew, done, info = self.venv.step(action)
        return obs[self.key], rew, done, info

    def close(self):
        return self.venv.close()


# Rango (r_min, r_max) de retorno por juego/dificultad para normalizar,
# igual tabla que eval_bct.py (Mediratta et al. ICLR 2024 / Cobbe et al.
# 2020 Procgen baselines).
PROCGEN = {
    "bigfish":    {"easy": (1, 40),    "hard": (0, 40)},
    "bossfight":  {"easy": (0.5, 13),  "hard": (0.5, 13)},
    "caveflyer":  {"easy": (3.5, 12),  "hard": (2, 13.4)},
    "chaser":     {"easy": (0.5, 13),  "hard": (0.5, 14.2)},
    "climber":    {"easy": (2, 12.6),  "hard": (1, 12.6)},
    "coinrun":    {"easy": (5, 10),    "hard": (5, 10)},
    "dodgeball":  {"easy": (1.5, 19),  "hard": (1.5, 19)},
    "fruitbot":   {"easy": (-1.5, 32.4), "hard": (-0.5, 27.2)},
    "heist":      {"easy": (3.5, 10),  "hard": (2, 10)},
    "jumper":     {"easy": (1, 10),    "hard": (1, 10)},
    "leaper":     {"easy": (1.5, 10),  "hard": (1.5, 10)},
    "maze":       {"easy": (5, 10),    "hard": (4, 10)},
    "miner":      {"easy": (1.5, 13),  "hard": (1.5, 20)},
    "ninja":      {"easy": (3.5, 10),  "hard": (2, 10)},
    "plunder":    {"easy": (4.5, 30),  "hard": (3, 30)},
    "starpilot":  {"easy": (2.5, 64),  "hard": (1.5, 35)},
}

# Splits fijos del protocolo (coinrun 1M expert, level_200): train = mismos
# niveles que vio el dataset offline, val = niveles nunca vistos de la
# misma familia, test = resto de la distribucion infinita de Procgen. Ver
# METODOLOGIA_DT_HDT.md, seccion 3.
SPLITS = {
    "train": dict(start_level=0, num_levels=200),
    "val": dict(start_level=200, num_levels=50),
    "test": dict(start_level=250, num_levels=0),
}


def normalize_return(raw_return, env_name, distribution_mode):
    r_min, r_max = PROCGEN[env_name][distribution_mode]
    return (raw_return - r_min) / (r_max - r_min)


def rollout(agent, env, traj_length, episode_length, target_return, max_steps, seed,
            frames=None):
    # Procgen no expone un seed() por episodio como mujoco-gym -- el
    # muestreo de nivel dentro de [start_level, start_level+num_levels) ya
    # lo controla ProcgenEnv internamente (num_levels=0 = distribucion
    # infinita). `seed` acá solo se usa para variar la semilla global entre
    # episodios (numpy), no hace falta pasarsela al entorno.
    np.random.seed(seed)
    obs = env.reset()  # (1, H, W, C)
    obs_frame = obs[0]

    obs_hist = deque(maxlen=traj_length)
    action_hist = deque(maxlen=traj_length)
    rtg_hist = deque(maxlen=traj_length)
    timestep_hist = deque(maxlen=traj_length)

    obs_hist.append(obs_frame)
    action_hist.append(np.zeros((1,), dtype=np.int64))
    rtg_hist.append(float(target_return))
    timestep_hist.append(0)

    total_reward = 0.0
    for t in range(max_steps):
        if frames is not None:
            frames.append(obs_hist[-1])
        obs_arr = np.stack(obs_hist)
        action_arr = np.stack(action_hist)
        rtg_arr = np.array(rtg_hist, dtype=np.float32).reshape(-1, 1)
        timestep_arr = np.array(timestep_hist, dtype=np.int64)

        with torch.no_grad():
            pred_action = agent.act(obs_arr, action_arr, rtg_arr, timestep_arr, step=t)
        action_idx = int(pred_action[0]) if np.ndim(pred_action) > 0 else int(pred_action)

        next_obs, reward, done, info = env.step(np.array([action_idx]))
        reward = float(reward[0])
        total_reward += reward

        action_hist[-1] = np.array([action_idx], dtype=np.int64)

        if done[0]:
            break

        next_timestep = min(timestep_hist[-1] + 1, episode_length - 1)
        obs_hist.append(next_obs[0])
        rtg_hist.append(rtg_hist[-1] - reward)
        action_hist.append(np.zeros((1,), dtype=np.int64))
        timestep_hist.append(next_timestep)

    return total_reward, t + 1


def eval_split(agent, traj_length, episode_length, target_return, max_steps,
                env_name, num_levels, start_level, distribution_mode, num_episodes, seed,
                video_path=None, video_episodes=0, video_fps=15, video_scale=8):
    env = procgen.ProcgenEnv(
        num_envs=1, env_name=env_name, num_levels=num_levels,
        start_level=start_level, distribution_mode=distribution_mode,
    )
    env = VecExtractDictObs(env, "rgb")

    agent.train(False)
    returns = []
    for ep in range(num_episodes):
        frames = [] if ep < video_episodes else None
        total_reward, n_steps = rollout(
            agent, env, traj_length, episode_length, target_return, max_steps, seed + ep,
            frames=frames,
        )
        returns.append(total_reward)
        if frames:
            # obs de 64x64 -> upscale por repeticion para que se vea en el video
            big = [np.kron(f, np.ones((video_scale, video_scale, 1), dtype=np.uint8))
                   for f in frames]
            out = video_path / f"ep{ep:02d}_ret{total_reward:.0f}.mp4"
            # imageio.mimsave falla en este env (imageio/imageio_ffmpeg desalineados)
            writer = imageio_ffmpeg.write_frames(
                str(out), big[0].shape[:2][::-1], fps=video_fps, macro_block_size=1
            )
            writer.send(None)
            for f in big:
                writer.send(np.ascontiguousarray(f))
            writer.close()
        print(f"  episodio {ep}: pasos={n_steps} retorno={total_reward:.2f}")
    env.close()
    return np.array(returns)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True, help="Path al snapshot_*.pt")
    parser.add_argument("--agent", choices=["dt", "hdt"], required=True)
    parser.add_argument("--env-name", default="coinrun")
    parser.add_argument("--distribution-mode", default="easy")
    parser.add_argument("--target-return", type=float, default=10.0,
                         help="Return-to-go objetivo. Default: 10.0 (retorno de nivel "
                              "completado en coinrun/easy -- reward es binario 0/10, "
                              "verificado sobre el dataset offline).")
    parser.add_argument("--num-episodes", type=int, default=100,
                         help="Episodios por split (100, igual que eval_bct.py / "
                              "Mediratta et al.)")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--episode-length", type=int, default=1000)
    parser.add_argument("--splits", nargs="+", default=list(SPLITS),
                         choices=list(SPLITS), help="Que splits correr (default: los 3)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--video-dir", default=None,
                         help="Si se pasa, guarda mp4 de los primeros episodios de cada "
                              "split en <video-dir>/<split>/")
    parser.add_argument("--video-episodes", type=int, default=3,
                         help="Episodios por split a grabar (solo con --video-dir)")
    parser.add_argument("--video-fps", type=int, default=15)
    args = parser.parse_args()

    payload = torch.load(args.snapshot, map_location=args.device)
    transformer_cfg = payload["cfg"]
    traj_length = transformer_cfg.traj_length
    num_actions = transformer_cfg.num_actions

    agent_cls = DTAgent if args.agent == "dt" else HDTAgent
    agent = agent_cls(
        name=f"{args.agent}_coinrun_eval",
        obs_shape=(64, 64, 3),
        action_shape=(num_actions,),
        device=args.device,
        lr=1e-4,
        batch_size=1,
        stddev_schedule=0.2,
        use_tb=False,
        transformer_cfg=transformer_cfg,
    )
    agent.model.load_state_dict(payload["model"], strict=False)
    agent.train(False)

    for split_name in args.splits:
        split = SPLITS[split_name]
        print(f"\n[{split_name}] start_level={split['start_level']} num_levels={split['num_levels']}")
        video_path = None
        if args.video_dir:
            video_path = Path(args.video_dir) / split_name
            video_path.mkdir(parents=True, exist_ok=True)
        returns = eval_split(
            agent, traj_length, args.episode_length, args.target_return, args.max_steps,
            args.env_name, split["num_levels"], split["start_level"], args.distribution_mode,
            args.num_episodes, args.seed,
            video_path=video_path,
            video_episodes=args.video_episodes if video_path else 0,
            video_fps=args.video_fps,
        )
        norm_scores = normalize_return(returns, args.env_name, args.distribution_mode)
        print(
            f"[{split_name}] ({args.agent}, target_return={args.target_return}): "
            f"retorno {returns.mean():.2f} +/- {returns.std():.2f} | "
            f"score normalizado {norm_scores.mean():.3f} +/- {norm_scores.std():.3f} "
            f"({args.num_episodes} episodios)"
        )


if __name__ == "__main__":
    main()
