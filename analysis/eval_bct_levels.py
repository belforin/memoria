"""
Per-level evaluation of a single BCT checkpoint over a range of ProcGen test
levels. For each level, records whether the agent won, its final return, and
the termination cause (win/lose/timeout) -- no trajectories are saved by
default, only the outcome summary per level. Run once per checkpoint; join
the resulting CSVs on `level` to build a level x model outcome map.

Every level is still pinned individually (num_levels=1, start_level=<level>
per rollout -- exactly one episode per level), but which levels get
evaluated can be chosen two ways:
  - fixed range (default): --start-level 300 --end-level 400 evaluates
    every level in [300, 400), in order.
  - unbounded random: --end-level 0 is a sentinel meaning "no upper bound,
    same as ProcGen's own num_levels=0 convention (see eval_bct.yaml's test
    split: start_level=250, num_levels=0)". In this mode --n-levels is
    required and that many DISTINCT levels are sampled uniformly at random
    from [start_level, 2**31 - 1] (procgen level_seed is int32) via
    random.Random(--level-seed).sample(...) -- pass --level-seed to make the
    draw replicable across runs (matches record_selected_levels.py's
    LEVEL_SELECTION_SEED pattern in rl-algo-impls); omit it for a fresh
    random draw each run.

Action selection is argmax (deterministic) by default; pass
--sample to switch to stochastic action selection
(Categorical(logits/temperature).sample()), matching the protocol used for
BCT eval during training (agent/mdp_bct.yaml default: sample=true,
temperature=1.0). Not seeded by default -- with --sample and no --seed,
re-running the same level range will not reproduce the same outcomes.

Two distinct step caps:
  --max-steps (default 1000): hard external safety valve for the rollout
    loop itself -- should basically never fire for coinrun (see below), kept
    as a guard against a hang on other games.
  --native-timeout-steps (default 500): the ProcGen engine's own per-episode
    cap for coinrun (confirmed via record_selected_levels.py in
    /home/bmancilla/archive/ProcGen/rl-algo-impls -- `done` fires natively at
    this length if the episode neither won nor died first; ProcGen's info
    dict carries no explicit timeout flag, so this is the only way to tell
    "ran out of time" apart from "died" once done=True with
    prev_level_complete=0). Adjust for other games' native step caps.

--record-dir enables saving full per-step trajectories (env_state,
obs_buffer, action_buffer, ...), one pickle per episode, in the exact format
used by record_selected_levels.py in rl-algo-impls (see that file's
docstring / the reference pkl at
/home/bmancilla/scratch/human_sequences/coinrun/states_human_l420.pkl):

  <record-dir>/<level-category>/states_<level-category>_seed<level>_rep00.pkl

  top-level dict: levels={level: steps}, arch, seed, checkpoint_step,
  env_seed, action_sample_seed, episode_return, episode_length, done_reason
  ("coin"/"death"/"timeout" -- coinrun vocabulary), level_category,
  requested_level_seed.
  each step in `steps`: t, env_state (raw procgen state bytes, pre-action),
  obs_buffer (list of up to K frames, last = frame at t), action_buffer
  (list of up to K-1 past actions, pre-decision), traj_action, source,
  reward.

Usage:
    python analysis/eval_bct_levels.py \\
        --path <snapshot.pt> --start-level 300 --end-level 400 \\
        --env-name coinrun --distribution-mode easy \\
        --out results_modelA.csv

    # stochastic, matching the training-time eval protocol:
    python analysis/eval_bct_levels.py \\
        --path <snapshot.pt> --start-level 300 --end-level 400 \\
        --sample --temperature 1.0 \\
        --out results_modelA_stochastic.csv

    # also record full trajectories:
    python analysis/eval_bct_levels.py \\
        --path <snapshot.pt> --start-level 300 --end-level 400 \\
        --out results_modelA.csv --record-dir seqs_modelA --level-category test

    # unbounded random, replicable via --level-seed (matches eval_bct.yaml's
    # test split start_level=250/num_levels=0):
    python analysis/eval_bct_levels.py \\
        --path <snapshot.pt> --start-level 250 --end-level 0 \\
        --n-levels 100 --level-seed 1 \\
        --out results_modelA_test_unbounded.csv
"""
import argparse
import csv
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import procgen
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.bc_ar import BCAREvalAgent
from analysis.trajectory_recorder import VecExtractDictObs, get_env_state

OBS_SHAPE = (64, 64, 3)
ACTION_SHAPE = (1,)
SOURCE_TAG = "bct_multimodal_eval"
MAX_LEVEL_SEED = 2 ** 31 - 1  # procgen level_seed is int32; upper bound for unbounded random mode


def select_levels(start_level, end_level, n_levels, level_seed):
    """
    end_level > start_level -> the fixed range [start_level, end_level), in order.
    end_level == 0          -> n_levels distinct levels sampled uniformly at random
                                from [start_level, MAX_LEVEL_SEED], via
                                random.Random(level_seed) (level_seed=None -> a
                                fresh, non-replicable draw each call).
    """
    if end_level == 0:
        assert n_levels is not None and n_levels > 0, (
            "--n-levels is required (and must be > 0) when --end-level=0 (unbounded random mode)"
        )
        rng = random.Random(level_seed)
        levels = rng.sample(range(start_level, MAX_LEVEL_SEED + 1), n_levels)
        levels.sort()
        return levels
    assert end_level > start_level, (
        "--end-level must be greater than --start-level, or exactly 0 for the unbounded random mode"
    )
    return list(range(start_level, end_level))


def eval_level(agent, env_name, distribution_mode, level, max_steps, native_timeout_steps, record):
    env = procgen.ProcgenEnv(
        num_envs=1, env_name=env_name, num_levels=1,
        start_level=level, distribution_mode=distribution_mode,
    )
    env = VecExtractDictObs(env, "rgb")

    agent.reset()
    obs = env.reset()
    done = False
    ep_return = 0.0
    steps_taken = 0
    infos = [{}]
    steps = [] if record else None

    while not done and steps_taken < max_steps:
        if record:
            env_state = get_env_state(env)
            action_buffer_snapshot = list(agent._action_buffer)

        action = agent.act(obs)

        if record:
            obs_buffer_snapshot = [f.copy() for f in agent._obs_buffer]

        obs, reward, done_arr, infos = env.step(action)
        done = bool(done_arr[0])
        reward_f = float(reward[0])
        ep_return += reward_f
        steps_taken += 1

        if record:
            steps.append({
                "t": steps_taken - 1,
                "env_state": env_state,
                "obs_buffer": obs_buffer_snapshot,
                "action_buffer": action_buffer_snapshot,
                "traj_action": int(action[0]),
                "source": SOURCE_TAG,
                "reward": reward_f,
            })

    env.close()

    if not done:
        cause = "timeout"                              # external --max-steps safety valve hit
    elif bool(infos[0].get("prev_level_complete", 0)):
        cause = "win"
    elif steps_taken >= native_timeout_steps:
        cause = "timeout"                              # native engine step cap reached without dying/winning
    else:
        cause = "lose"

    result = {
        "level": level,
        "won": cause == "win",
        "return": ep_return,
        "termination_cause": cause,
        "steps": steps_taken,
    }
    return result, steps


def save_episode_pkl(record_dir, level_category, level, steps, result, arch, checkpoint_step, seed):
    done_reason = {"win": "coin", "lose": "death", "timeout": "timeout"}[result["termination_cause"]]
    category_dir = Path(record_dir) / level_category
    category_dir.mkdir(parents=True, exist_ok=True)
    out_path = category_dir / f"states_{level_category}_seed{level}_rep00.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({
            "levels": {level: steps},
            "arch": arch,
            "seed": seed,
            "checkpoint_step": checkpoint_step,
            "env_seed": seed,
            "action_sample_seed": seed,
            "episode_return": result["return"],
            "episode_length": result["steps"],
            "done_reason": done_reason,
            "level_category": level_category,
            "requested_level_seed": level,
        }, f)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--path", required=True, help="snapshot_*.pt to load the BCT agent's weights from")
    parser.add_argument("--env-name", default="coinrun")
    parser.add_argument("--distribution-mode", default="easy")
    parser.add_argument("--start-level", type=int, required=True, help="first level to evaluate (inclusive)")
    parser.add_argument("--end-level", type=int, required=True,
                         help="last level to evaluate (exclusive) -- evaluates range(start_level, end_level). "
                              "Pass 0 for unbounded random mode: sample --n-levels distinct levels uniformly at "
                              "random from [start_level, 2**31-1] instead (see --n-levels/--level-seed).")
    parser.add_argument("--n-levels", type=int, default=None,
                         help="how many distinct levels to sample when --end-level=0 (required in that mode; "
                              "unused otherwise)")
    parser.add_argument("--level-seed", type=int, default=None,
                         help="seed for the random level draw when --end-level=0, for a replicable selection "
                              "across runs. Leave unset for a fresh random draw each run.")
    parser.add_argument("--max-steps", type=int, default=1000,
                         help="hard external safety cap for the rollout loop; should not normally fire for coinrun")
    parser.add_argument("--native-timeout-steps", type=int, default=500,
                         help="ProcGen engine's native per-episode step cap (500 for coinrun) -- used to tell "
                              "a native timeout apart from a death once done=True with prev_level_complete=0")
    parser.add_argument("--sample", action="store_true",
                         help="stochastic action selection (Categorical(logits/temperature).sample()) instead of "
                              "argmax -- matches the training-time BCT eval protocol. Off by default (deterministic "
                              "argmax).")
    parser.add_argument("--temperature", type=float, default=1.0, help="sampling temperature, only used with --sample")
    parser.add_argument("--seed", type=int, default=None,
                         help="torch RNG seed, applied once before the eval loop (only matters with --sample). "
                              "Also stored as seed/action_sample_seed/env_seed in recorded pkls for provenance. "
                              "Leave unset for non-reproducible stochastic runs.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True, help="output CSV path")
    parser.add_argument("--record-dir", default=None,
                         help="if set, also saves one pkl per episode with full trajectories (env_state, "
                              "obs_buffer, action_buffer, ...) in rl-algo-impls's record_selected_levels.py format")
    parser.add_argument("--level-category", default="test",
                         help="label stored in each pkl's level_category field and used as the pkl output subdir "
                              "(e.g. test/val/train) -- only relevant with --record-dir")
    parser.add_argument("--arch", default="bct_multimodal",
                         help="tag stored in each pkl's arch field, identifying this model -- only relevant with --record-dir")
    args = parser.parse_args()

    levels = select_levels(args.start_level, args.end_level, args.n_levels, args.level_seed)
    if args.end_level == 0:
        print(f"[eval_bct_levels] modo aleatorio no acotado: {len(levels)} niveles sorteados desde "
              f"start_level={args.start_level} con level_seed={args.level_seed} -> {levels}")

    if args.seed is not None:
        torch.manual_seed(args.seed)

    try:
        checkpoint_step = int(Path(args.path).stem.split("_")[-1])
    except ValueError:
        checkpoint_step = None

    device = torch.device(args.device)
    agent = BCAREvalAgent(
        obs_shape=OBS_SHAPE, action_shape=ACTION_SHAPE, device=device,
        temperature=args.temperature, sample=args.sample, path=args.path,
    )

    results = []
    for level in levels:
        result, steps = eval_level(
            agent, args.env_name, args.distribution_mode, level,
            args.max_steps, args.native_timeout_steps, record=args.record_dir is not None,
        )
        results.append(result)
        print(f"[eval_bct_levels] level={result['level']:>6d} | "
              f"{result['termination_cause']:>7s} | return={result['return']:.2f} | steps={result['steps']}")

        if args.record_dir is not None:
            pkl_path = save_episode_pkl(
                args.record_dir, args.level_category, level, steps, result,
                args.arch, checkpoint_step, args.seed,
            )
            print(f"[eval_bct_levels]   -> {pkl_path}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["level", "won", "return", "termination_cause", "steps"])
        writer.writeheader()
        writer.writerows(results)

    n_win = sum(r["won"] for r in results)
    n_lose = sum(r["termination_cause"] == "lose" for r in results)
    n_timeout = sum(r["termination_cause"] == "timeout" for r in results)
    mean_return = float(np.mean([r["return"] for r in results]))
    level_range_desc = (
        f"{len(levels)} random from {args.start_level}+ (level_seed={args.level_seed})"
        if args.end_level == 0 else f"{args.start_level}..{args.end_level - 1}"
    )
    print(f"\n[eval_bct_levels] {len(results)} levels evaluated "
          f"({level_range_desc}) | win={n_win} lose={n_lose} timeout={n_timeout} "
          f"| mean_return={mean_return:.2f}")
    print(f"[eval_bct_levels] -> {out_path}")


if __name__ == "__main__":
    main()
