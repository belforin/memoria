"""
Convierte un dataset D4RL (formato hdf5 estándar de gym-mujoco: observations,
actions, rewards, terminals, timeouts, next_observations) al formato de
episodios .npz que espera replay_buffer.OfflineReplayBuffer.

Cada episodio de largo T se guarda con arrays de largo T+1 (transición
"dummy" inicial + T transiciones reales), siguiendo la misma convención que
usa el resto del repo (ver replay_buffer.episode_len / _sample):
    observation[0]   = obs inicial real del episodio
    observation[i]   = obs luego de aplicar action[i], para i=1..T
    action[0]        = dummy (no se usa nunca como acción real)
    action[i]        = accion tomada en el paso i-1 del episodio original
    reward[0]        = dummy (0)
    reward[i]        = reward recibido en el paso i-1
    discount[0]       = dummy (1)
    discount[i]       = 1 - terminal en el paso i-1

No requiere el paquete `d4rl` ni `mujoco_py` (que no cargan en esta
maquina); solo h5py + numpy, ya instalados en maskdp-env.
"""
import argparse
from pathlib import Path

import h5py
import numpy as np

from replay_buffer import save_episode


def convert(hdf5_path, out_dir, domain):
    out_dir = Path(out_dir) / domain
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(hdf5_path, "r") as f:
        observations = f["observations"][:]
        actions = f["actions"][:]
        rewards = f["rewards"][:]
        terminals = f["terminals"][:]
        timeouts = f["timeouts"][:] if "timeouts" in f else np.zeros_like(terminals)
        next_observations = f["next_observations"][:]

    obs_dim = observations.shape[1]
    action_dim = actions.shape[1]

    done = terminals | timeouts
    end_idxs = np.where(done)[0]

    n_episodes = 0
    start = 0
    for end in end_idxs:
        T = end - start + 1
        ep_obs = np.concatenate(
            [observations[start : end + 1], next_observations[end : end + 1]], axis=0
        )  # (T+1, obs_dim)
        ep_action = np.concatenate(
            [np.zeros((1, action_dim), dtype=actions.dtype), actions[start : end + 1]],
            axis=0,
        )  # (T+1, action_dim)
        ep_reward = np.concatenate(
            [[0.0], rewards[start : end + 1]]
        ).astype(np.float32).reshape(T + 1, 1)
        ep_discount = np.concatenate(
            [[1.0], 1.0 - terminals[start : end + 1].astype(np.float32)]
        ).astype(np.float32).reshape(T + 1, 1)

        episode = {
            "observation": ep_obs.astype(np.float32),
            "action": ep_action.astype(np.float32),
            "reward": ep_reward,
            "discount": ep_discount,
        }
        fn = out_dir / f"episode_{n_episodes}_{T}.npz"
        save_episode(episode, fn)
        n_episodes += 1
        start = end + 1

    print(
        f"{n_episodes} episodios guardados en {out_dir} "
        f"(obs_dim={obs_dim}, action_dim={action_dim})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", required=True, help="Path al .hdf5 de D4RL")
    parser.add_argument("--out", required=True, help="Directorio de salida (padre de <domain>/)")
    parser.add_argument("--domain", required=True, help="Nombre de dominio, ej. halfcheetah")
    args = parser.parse_args()
    convert(args.hdf5, args.out, args.domain)
