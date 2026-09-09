"""
Entrypoint de pretraining para agent=dt / agent=hdt contra datos offline de
D4RL (convertidos previamente con d4rl_data.py). A diferencia de
pretrain.py, no depende de dmc/dm_control: obs_shape/action_shape vienen de
la config, y el replay buffer se arma con env=None + relabel=False (dt/hdt
nunca relabelean reward contra un env en vivo).
"""
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

from pathlib import Path

import hydra
import numpy as np
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


def compute_obs_stats(replay_dir, domain):
    """
    Media y desviacion estandar de las observaciones del dataset de
    entrenamiento, para normalizar antes de la proyeccion de estado en
    DT/HDT -- paso del codigo oficial de Decision Transformer
    (kzl/decision-transformer, gym/experiment.py) que faltaba en nuestro
    pipeline (nunca normalizabamos obs, ni en entrenamiento ni en
    evaluacion). Ver METODOLOGIA_DT_HDT.md, seccion 2.8.
    """
    obs = []
    for fn in sorted((Path(replay_dir) / domain).rglob("*.npz")):
        with fn.open("rb") as f:
            episode = np.load(f)
            # excluye el ultimo estado (bootstrap next_obs), igual que
            # OfflineReplayBuffer._sample no lo usa como "obs" de entrada
            obs.append(episode["observation"][:-1])
    obs = np.concatenate(obs, axis=0)
    mean = obs.mean(axis=0)
    std = obs.std(axis=0) + 1e-6
    return mean.astype(np.float32), std.astype(np.float32)


@hydra.main(config_path=".", config_name="pretrain_dt")
def main(cfg):
    work_dir = Path.cwd()
    print(f"workspace: {work_dir}")

    utils.set_seed_everywhere(cfg.seed)
    device = torch.device(cfg.device)

    obs_shape = (cfg.obs_dim,)
    action_shape = (cfg.action_dim,)
    domain = get_domain(cfg.task)

    obs_mean, obs_std = compute_obs_stats(cfg.replay_buffer_dir, domain)
    print(f"obs stats ({domain}): mean={obs_mean}, std={obs_std}")

    # create agent
    agent = hydra.utils.instantiate(
        cfg.agent,
        obs_shape=obs_shape,
        action_shape=action_shape,
        obs_mean=obs_mean,
        obs_std=obs_std,
    )

    if cfg.resume is True:
        resume_dir = get_dir(cfg)
        payload = torch.load(resume_dir)
        agent.model.load_state_dict(payload["model"], strict=False)

    snapshot_dir = work_dir / Path(cfg.snapshot_dir) / domain / str(cfg.seed)
    snapshot_dir.mkdir(exist_ok=True, parents=True)

    # create logger
    cfg.agent.obs_shape = obs_shape
    cfg.agent.action_shape = action_shape
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
