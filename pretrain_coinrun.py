"""
Entrypoint de pretraining para agent=dt_coinrun / agent=hdt_coinrun contra
el dataset offline de CoinRun (convertido previamente con
convert_coinrun_to_npz.py). Copia adaptada de pretrain_dt.py para la Etapa 2
(visual) -- ver METODOLOGIA_DT_HDT.md, seccion 3.

Diferencias con pretrain_dt.py:
- obs_shape es fijo (64,64,3) -- no hay "obs_dim" escalar para pixeles.
- action_shape sale de agent.transformer_cfg.num_actions (Procgen: 15), no
  de un cfg.action_dim aparte -- unica fuente de verdad, igual que
  eval_coinrun.py.
- Sin compute_obs_stats: no aplica a pixeles (DecisionTransformer/
  HierarchicalDecisionTransformer ignoran obs_mean/obs_std cuando
  pixel_obs=True, ver agent/dt.py). No se pasa obs_mean/obs_std al agente.
"""
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

from pathlib import Path

import hydra
import torch
import omegaconf

import utils
from logger import Logger
from replay_buffer import make_replay_loader
import wandb

torch.backends.cudnn.benchmark = True


def get_dir(cfg):
    resume_dir = Path(cfg.resume_dir)
    snapshot = resume_dir / str(cfg.seed) / f"snapshot_{cfg.resume_step}.pt"
    print("loading from", snapshot)
    return snapshot


def get_domain(task):
    return task.split("_", 1)[0]


@hydra.main(config_path=".", config_name="pretrain_coinrun")
def main(cfg):
    work_dir = Path.cwd()
    print(f"workspace: {work_dir}")

    utils.set_seed_everywhere(cfg.seed)
    device = torch.device(cfg.device)

    obs_shape = tuple(cfg.obs_shape)
    action_shape = (cfg.agent.transformer_cfg.num_actions,)
    domain = get_domain(cfg.task)

    # create agent
    agent = hydra.utils.instantiate(
        cfg.agent,
        obs_shape=obs_shape,
        action_shape=action_shape,
    )

    if cfg.resume is True:
        resume_dir = get_dir(cfg)
        payload = torch.load(resume_dir)
        agent.model.load_state_dict(payload["model"], strict=False)

    snapshot_dir = work_dir / Path(cfg.snapshot_dir) / domain / str(cfg.seed)
    snapshot_dir.mkdir(exist_ok=True, parents=True)

    # create logger
    cfg.agent.obs_shape = list(obs_shape)
    cfg.agent.action_shape = list(action_shape)
    exp_name = "_".join([cfg.agent.name, domain, str(cfg.seed)])
    wandb_config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    wandb.init(
        project=cfg.project,
        entity="maskdp",
        name=exp_name,
        config=wandb_config,
        settings=wandb.Settings(
            start_method="thread",
            _disable_stats=True,
        ),
        mode="online" if cfg.use_wandb else "offline",
        notes=cfg.notes,
    )
    logger = Logger(work_dir, use_tb=cfg.use_tb, use_wandb=cfg.use_wandb)

    replay_train_dir = Path(cfg.replay_buffer_dir) / domain
    print(f"replay dir: {replay_train_dir}")
    train_loader = make_replay_loader(
        None,
        replay_train_dir,
        cfg.replay_buffer_size,
        cfg.batch_size,
        cfg.replay_buffer_num_workers,
        cfg.discount,
        domain,
        cfg.agent.transformer_cfg.traj_length,
        relabel=False,
        return_to_go=True,
    )
    train_iter = iter(train_loader)

    timer = utils.Timer()

    global_step = cfg.resume_step

    train_until_step = utils.Until(cfg.num_grad_steps)
    log_every_step = utils.Every(cfg.log_every_steps)

    while train_until_step(global_step):
        metrics = agent.update(train_iter, global_step)
        logger.log_metrics(metrics, global_step, ty="train")
        if log_every_step(global_step):
            elapsed_time, total_time = timer.reset()
            with logger.log_and_dump_ctx(global_step, ty="train") as log:
                log("fps", cfg.log_every_steps / elapsed_time)
                log("total_time", total_time)
                log("step", global_step)

        if global_step in cfg.snapshots:
            snapshot = snapshot_dir / f"snapshot_{global_step}.pt"
            payload = {
                "model": agent.model.state_dict(),
                "cfg": cfg.agent.transformer_cfg,
            }
            with snapshot.open("wb") as f:
                torch.save(payload, f)

        global_step += 1


if __name__ == "__main__":
    main()
