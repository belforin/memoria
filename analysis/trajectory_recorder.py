import csv
import pickle
from collections import deque
from pathlib import Path

import numpy as np
import procgen


class VecExtractDictObs:

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


def _unwrap_procgen(env):
    """
    Unwraps the environment to find the underlying object that exposes
    `get_state()` and `set_state()`. Fails loudly if neither method is found
    in the known wrapper chain.
    """
    candidates = [env.venv, getattr(env.venv, "env", None), getattr(env.venv, "gym3_env", None)]
    for c in candidates:
        if c is not None and hasattr(c, "get_state") and hasattr(c, "set_state"):
            return c
    raise RuntimeError(
        "Could not find get_state()/set_state() in the wrapper chain. "
        "Checked candidates: env.venv, env.venv.env, env.venv.gym3_env. "
        "Check your procgen version and adjust _unwrap_procgen()."
    )


def get_env_state(env):
    raw = _unwrap_procgen(env)
    return raw.get_state()[0]  # num_envs=1 -> list[bytes] of length 1


def set_env_state(env, state_bytes):
    raw = _unwrap_procgen(env)
    raw.set_state([state_bytes])


def record_reference_trajectory(agent, env_name, level, distribution_mode):
    """
    Performs a real rollout of `agent` on a single fixed level and saves it 
    in a reusable format. It is architecture-agnostic, working with any agent 
    that exposes `.reset()`, `.act(obs)`, and `.sample`.

    Records the exact environment state and buffer states at each step so
    the exact context can be reproduced later for evaluation.

    Buffer capture order: `action_buffer` is captured BEFORE calling `act()`
    (pre-decision state), and `obs_buffer` is captured AFTER (to include
    the newly observed frame).

    Requires `agent.sample=False` (argmax) for exploitation, not exploration.
    """
    assert not agent.sample, "record_reference_trajectory requires agent.sample=False (argmax)."

    env = procgen.ProcgenEnv(
        num_envs=1, env_name=env_name, num_levels=1,
        start_level=level, distribution_mode=distribution_mode,
    )
    env = VecExtractDictObs(env, "rgb")

    agent.reset()
    obs = env.reset()
    done = False
    t = 0
    steps = []

    while not done:
        env_state = get_env_state(env)
        action_buffer_snapshot = list(agent._action_buffer)          # pre-decision
        action = agent.act(obs)                                      # decide + append
        obs_buffer_snapshot = [f.copy() for f in agent._obs_buffer]  # post-append

        steps.append({
            "t": t,
            "env_state": env_state,
            "obs_buffer": obs_buffer_snapshot,
            "action_buffer": action_buffer_snapshot,
            "traj_action": int(action[0]),
        })

        obs, reward, done, infos = env.step(action)
        t += 1

    env.close()
    return steps


def quantile_indices(episode_len, num_states):
    if episode_len <= num_states:
        return list(range(episode_len))
    fracs = np.linspace(0.05, 0.95, num_states)
    idx = np.unique(np.round(fracs * (episode_len - 1)).astype(int))
    return idx.tolist()


def select_quantile_states(steps, num_states):
    idx = quantile_indices(len(steps), num_states)
    return [steps[i] for i in idx]


def save_trajectory_states(out_root, name, arch, seed, checkpoint_step, trajectories):
    """`trajectories`: {level: [state, ...]}. Returns the written path."""
    states_path = Path(out_root) / f"states_{name}.pkl"
    with open(states_path, "wb") as f:
        pickle.dump({
            "levels": trajectories,
            "arch": arch,
            "seed": seed,
            "checkpoint_step": checkpoint_step,
        }, f)
    return states_path


def load_trajectory_states(out_root, name):
    path = Path(out_root) / f"states_{name}.pkl"
    assert path.exists(), f"Missing {path}. Record a trajectory for '{name}' first."
    with open(path, "rb") as f:
        return pickle.load(f)["levels"]


def discover_state_pkl_names(states_dir):
    """
    Autodiscovers all `states_<name>.pkl` in `states_dir` -- the one-pkl-per-episode
    schema written by analysis/eval_bct_levels.py (NOT the one-pkl-per-model schema
    written by `save_trajectory_states` above; both are read by `load_trajectory_states`).
    Same discovery rule as run_attn_attr_batch.py's `_discover_names`.
    """
    states_dir = Path(states_dir)
    names = sorted(p.name[len("states_"):-len(".pkl")] for p in states_dir.glob("states_*.pkl"))
    assert names, f"no states_*.pkl found in {states_dir}"
    return names


def load_attattr_index(index_csv):
    """
    Reads an index.csv written by run_attn_attr_batch.py and returns the set of
    (name, level, state_index) triples it successfully attributed (error == "").
    `state_index` is the position of the state within `by_level[level]` for that
    pkl -- `idx` in `enumerate(by_level[level])`, exactly how run_attn_attr_batch.py
    keys its own work items.
    """
    keep = set()
    with open(index_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row["error"] == "":
                keep.add((row["name"], int(row["level"]), int(row["state_index"])))
    assert keep, f"no successful rows (error=='') found in {index_csv}"
    return keep


def load_states_matching_attattr(states_dir, index_csv, levels=None, states_per_episode=None):
    """
    Loads states_*.pkl from `states_dir` (ATTATTR's per-episode schema) and
    restricts them to exactly the states ATTATTR successfully attributed, per
    `index_csv` -- giving an exact 1:1 state set between SARFA and ATTATTR for
    this model (same (level, t) states, same recorded env/buffer snapshots).
    This is the join key, not position/order within a pkl's own state list.

    `levels`: optional iterable to further restrict to a subset of levels
    (None -> keep every level present in `index_csv`).

    `states_per_episode`: optional int. If given, each level's ATTATTR-matched
    states are further subsampled via `select_quantile_states` (quantile
    spacing over the matched, t-ordered list) -- this is a subset of what
    ATTATTR attributed, not a 1:1 match against everything. None (default) ->
    keep every ATTATTR-matched state.

    Returns {level: [state, ...]}, merged across every pkl in `states_dir`.
    """
    names = discover_state_pkl_names(states_dir)
    keep = load_attattr_index(index_csv)
    level_filter = set(levels) if levels else None

    merged = {}
    n_kept = 0
    for name in names:
        by_level = load_trajectory_states(states_dir, name)
        for level, states in by_level.items():
            if level_filter is not None and level not in level_filter:
                continue
            matched = [state for idx, state in enumerate(states) if (name, level, idx) in keep]
            n_kept += len(matched)
            merged.setdefault(level, []).extend(matched)

    n_expected = len(keep) if level_filter is None else sum(
        1 for (_, level, _) in keep if level in level_filter
    )
    assert n_kept == n_expected, (
        f"matched {n_kept}/{n_expected} states from {index_csv} against the pkls "
        f"in {states_dir} -- some (name, level, state_index) rows had no matching "
        f"state. Check that states_dir and index_csv come from the same ATTATTR run."
    )

    if states_per_episode is not None:
        merged = {level: select_quantile_states(states, states_per_episode) for level, states in merged.items()}

    return merged


def load_buffers(agent, obs_buffer_snapshot, action_buffer_snapshot):
    """
    Loads a saved snapshot back into the agent's buffers for teacher forcing. 
    The agent does not need to be the one that originally recorded it.
    """
    assert len(obs_buffer_snapshot) <= agent.K, (
        f"obs_buffer_snapshot has {len(obs_buffer_snapshot)} frames but "
        f"agent.K={agent.K}. The agent replaying this snapshot must share "
        f"the K it was recorded with."
    )
    agent._obs_buffer = deque(obs_buffer_snapshot, maxlen=agent.K)
    agent._action_buffer = deque(action_buffer_snapshot, maxlen=agent.K - 1)
