import hydra
import torch
from pathlib import Path
import dmc
from replay_buffer import make_replay_loader


@hydra.main(config_path=".", config_name="pretrain")
def main(cfg):
    device = torch.device(cfg.device)
    env = dmc.make(cfg.task, seed=cfg.seed)
    domain = cfg.task.split("_", 1)[0]

    replay_train_dir = Path(cfg.replay_buffer_dir) / domain
    loader = make_replay_loader(
        env,
        replay_train_dir,
        cfg.replay_buffer_size,
        cfg.batch_size,
        cfg.replay_buffer_num_workers,
        cfg.discount,
        domain,
        cfg.agent.transformer_cfg.traj_length,
        relabel=False,
    )
    train_iter = iter(loader)
    batch = next(train_iter)  # mismo batch para todos los snapshots -> comparación justa

    snap_dir = Path(cfg.snapshot_dir) / domain / str(cfg.seed)
    names = ["snapshot_0.pt", "snapshot_100000.pt", "snapshot_200000.pt",
             "snapshot_300000.pt", "snapshot_350000.pt", "snapshot_400000.pt"]

    print(f"{'snapshot':<22} {'state_loss':>12} {'action_loss':>12}")
    for name in names:
        path = snap_dir / name
        if not path.exists():
            print(f"{name:<22} (no existe, salto)")
            continue

        # instancia el agente normal (pesos random), luego pisa los pesos con el snapshot
        agent = hydra.utils.instantiate(
            cfg.agent,
            obs_shape=env.observation_spec().shape,
            action_shape=env.action_spec().shape,
        )
        payload = torch.load(path, map_location=device)
        agent.model.load_state_dict(payload["model"])

        agent.use_tb = True  # fuerza a que update() calcule y devuelva las pérdidas
        metrics = agent.update(iter([batch]), 0)
        sl = metrics.get("state_loss", float("nan"))
        al = metrics.get("action_loss", float("nan"))
        print(f"{name:<22} {sl:>12.6f} {al:>12.6f}")


if __name__ == "__main__":
    main()
