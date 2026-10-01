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

Etapa BC (METODOLOGIA_DT_HDT.md seccion 7): agentes, semillas y ruta de
snapshots configurables (--agents/--seeds/--snapshot-pattern), varios
rand_seed de evaluacion (--eval-seeds) para separar la varianza entre
semillas de entrenamiento del ruido por muestreo de niveles, y muestreo
con temperatura (--sample/--temperature) como eval_bct.yaml de Benjamin.
Los defaults reproducen la evaluacion rtgfix anterior.

Requiere el env `procgen-env`, igual que eval_coinrun.py.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from agent.dt import DTAgent
from agent.hdt import HDTAgent
from eval_coinrun import (
    SPLITS, add_sampling_args, eval_split, normalize_return, set_sampling,
)

DEFAULT_PATTERN = "~/snapshot/coinrun_{agent}_rtgfix/coinrun/{seed}/snapshot_{step}.pt"


def snapshot_path(pattern, agent_name, seed, step):
    return Path(pattern.format(agent=agent_name, seed=seed, step=step)).expanduser()


def agent_class(agent_name):
    # topologia jerarquica: hdt (con rtg) o bc_hier (sin rtg)
    return HDTAgent if ("hdt" in agent_name or "hier" in agent_name) else DTAgent


def load_agent(snapshot, agent_name, device):
    payload = torch.load(snapshot, map_location=device)
    transformer_cfg = payload["cfg"]
    num_actions = transformer_cfg.num_actions

    agent = agent_class(agent_name)(
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
    parser.add_argument("--agents", nargs="+", default=["dt", "hdt"],
                         help="Nombres que se reemplazan en {agent} de --snapshot-pattern; "
                              "'hdt'/'hier' en el nombre usa HDTAgent")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3],
                         help="Semillas de entrenamiento")
    parser.add_argument("--snapshot-pattern", default=DEFAULT_PATTERN)
    parser.add_argument("--env-name", default="coinrun")
    parser.add_argument("--distribution-mode", default="easy")
    parser.add_argument("--target-return", type=float, default=10.0,
                         help="Solo para agentes con rtg; los BC sin rtg lo ignoran")
    parser.add_argument("--num-episodes", type=int, default=100,
                         help="Episodios por split, semilla y rand_seed (100, igual que eval_coinrun.py)")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--episode-length", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0,
                         help="rand_seed de evaluacion si no se pasa --eval-seeds")
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=None,
                         help="Varios rand_seed de evaluacion (p. ej. 0 1 2 3 4)")
    add_sampling_args(parser)
    args = parser.parse_args()
    eval_seeds = args.eval_seeds or [args.seed]

    print(f"modo: {'sample T=%g' % args.temperature if args.sample else 'argmax'}, "
          f"eval rand_seeds={eval_seeds}, {args.num_episodes} episodios cada uno")
    print(f"{'agente':<8} {'split':<6} " + " ".join(f"{'s%d' % s:>7}" for s in args.seeds)
          + "   media +/- std (entre semillas) | std entre rand_seeds (media)")

    all_results = {}
    for agent_name in args.agents:
        for split_name, split in SPLITS.items():
            seed_scores = []
            eval_stds = []
            per_seed_cells = []
            for seed in args.seeds:
                path = snapshot_path(args.snapshot_pattern, agent_name, seed, args.step)
                if not path.exists():
                    per_seed_cells.append("falta")
                    continue
                agent, traj_length = load_agent(path, agent_name, args.device)
                set_sampling(agent, args)
                scores = []
                for eval_seed in eval_seeds:
                    returns = eval_split(
                        agent, traj_length, args.episode_length, args.target_return,
                        args.max_steps, args.env_name, split["num_levels"], split["start_level"],
                        args.distribution_mode, args.num_episodes, eval_seed,
                    )
                    scores.append(
                        normalize_return(returns, args.env_name, args.distribution_mode).mean()
                    )
                seed_scores.append(float(np.mean(scores)))
                eval_stds.append(float(np.std(scores)))
                per_seed_cells.append(f"{np.mean(scores):.3f}")

            if seed_scores:
                agg_mean = float(np.mean(seed_scores))
                agg_std = float(np.std(seed_scores))
                eval_std = float(np.mean(eval_stds))
                all_results[(agent_name, split_name)] = (agg_mean, agg_std, eval_std, len(seed_scores))
                agg_str = (f"{agg_mean:.3f} +/- {agg_std:.3f} (n={len(seed_scores)}) | "
                           f"{eval_std:.3f}")
            else:
                agg_str = "sin snapshots"

            cells = " ".join(f"{c:>7}" for c in per_seed_cells)
            print(f"{agent_name:<8} {split_name:<6} {cells}   {agg_str}", flush=True)

    return all_results


if __name__ == "__main__":
    main()
