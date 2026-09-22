"""
Agrega eval_coinrun.py sobre las 3 semillas del re-entrenamiento CoinRun con
rtg corregido (METODOLOGIA_DT_HDT.md, seccion 2.10/2.12, punto 12 del
resumen de proximos pasos: jobs 28685/28686 semilla 1, 28845-28848 semillas
2/3) para poder decidir si la diferencia de generalizacion DT vs. HDT
encontrada con una sola semilla es señal real o varianza de entrenamiento --
mismo objetivo que eval_seeds.py para D4RL, pero para CoinRun.

eval_coinrun.py ya promedia --num-episodes episodios de UNA corrida sobre
UN split; este script corre eso para cada semilla {1,2,3} de cada
(agente, split) y reporta media +/- desviacion ENTRE SEMILLAS del score
normalizado, ademas de las 3 semillas individuales.

No graba video (ya existen para la semilla 1 en
eval_results/videos_coinrun_{dt,hdt}_rtgfix_100000/) -- este script es solo
para el numero agregado.

Requiere el env `procgen-env`, igual que eval_coinrun.py.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from agent.dt import DTAgent
from agent.hdt import HDTAgent
from eval_coinrun import SPLITS, eval_split, normalize_return

SEEDS = [1, 2, 3]
AGENTS = ["dt", "hdt"]


def snapshot_path(agent_name, seed, step):
    return (
        Path.home() / "snapshot" / f"coinrun_{agent_name}_rtgfix"
        / "coinrun" / str(seed) / f"snapshot_{step}.pt"
    )


def load_agent(snapshot, agent_name, device):
    payload = torch.load(snapshot, map_location=device)
    transformer_cfg = payload["cfg"]
    num_actions = transformer_cfg.num_actions

    agent_cls = DTAgent if agent_name == "dt" else HDTAgent
    agent = agent_cls(
        name=f"{agent_name}_coinrun_seeds_eval",
        obs_shape=(64, 64, 3),
        action_shape=(num_actions,),
        device=device,
        lr=1e-4,
        batch_size=1,
        stddev_schedule=0.2,
        use_tb=False,
        transformer_cfg=transformer_cfg,
    )
    agent.model.load_state_dict(payload["model"], strict=False)
    agent.train(False)
    return agent, transformer_cfg.traj_length


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=100000)
    parser.add_argument("--env-name", default="coinrun")
    parser.add_argument("--distribution-mode", default="easy")
    parser.add_argument("--target-return", type=float, default=10.0)
    parser.add_argument("--num-episodes", type=int, default=100,
                         help="Episodios por split y semilla (100, igual que eval_coinrun.py)")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--episode-length", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"{'agente':<6} {'split':<6} {'s1':>7} {'s2':>7} {'s3':>7}   media +/- std (semillas)")

    all_results = {}
    for agent_name in AGENTS:
        for split_name, split in SPLITS.items():
            seed_scores = []
            per_seed_cells = []
            for seed in SEEDS:
                path = snapshot_path(agent_name, seed, args.step)
                if not path.exists():
                    per_seed_cells.append("falta")
                    continue
                agent, traj_length = load_agent(path, agent_name, args.device)
                returns = eval_split(
                    agent, traj_length, args.episode_length, args.target_return,
                    args.max_steps, args.env_name, split["num_levels"], split["start_level"],
                    args.distribution_mode, args.num_episodes, args.seed,
                )
                score = normalize_return(returns, args.env_name, args.distribution_mode).mean()
                seed_scores.append(score)
                per_seed_cells.append(f"{score:.3f}")

            if seed_scores:
                agg_mean = float(np.mean(seed_scores))
                agg_std = float(np.std(seed_scores))
                all_results[(agent_name, split_name)] = (agg_mean, agg_std, len(seed_scores))
                agg_str = f"{agg_mean:.3f} +/- {agg_std:.3f} (n={len(seed_scores)})"
            else:
                agg_str = "sin snapshots"

            cells = "  ".join(f"{c:>7}" for c in per_seed_cells)
            print(f"{agent_name:<6} {split_name:<6} {cells}   {agg_str}")

    return all_results


if __name__ == "__main__":
    main()
