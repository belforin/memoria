"""
Convierte el dataset crudo de CoinRun (gen_dgrl, un .npy por trayectoria) al
formato .npz que ya consume replay_buffer.py / OfflineReplayBuffer (el mismo
que usa halfcheetah/hopper/walker2d hoy). NO modifica replay_buffer.py ni
nada del modelo: es un paso de preprocesamiento de datos, que se corre UNA
VEZ, separado del entrenamiento.

Adaptado del script equivalente de Benjamin Mancilla
(MaskDP_mmodal/hier-procgen), reutilizado tal cual para la Etapa 2 (regimen
visual) -- ver METODOLOGIA_DT_HDT.md, seccion 3.

Contrato de ENTRADA (verificado empiricamente sobre coinrun-1M_E, fuente
gen_dgrl/torchrl: https://github.com/facebookresearch/gen_dgrl):
    observations: (T+1, 3, 64, 64)  uint8,  rango [0,255], CHW
    actions:      (T,   1)          int64,  valores en [0,14]
    rewards:      (T,   1)          float32
    dones:        (T,)              float32, ultimo valor = 1.0

Contrato de SALIDA (igual al que replay_buffer.py ya espera):
    observation: (T+1, 64, 64, 3)   uint8,  HWC   <- transpuesto desde CHW
    action:      (T+1, 1)           int64         <- dummy en idx 0
    reward:      (T+1, 1)           float32       <- dummy en idx 0
    discount:    (T+1, 1)           float32       <- sintetico, todo unos
    done:        (T+1,)             float32       <- dummy en idx 0 (no la usa
                                                       _sample() hoy, se preserva
                                                       igual, no cuesta nada)

Politica de episodios cortos (decision explicita, no implicita):
    OfflineReplayBuffer._sample() hace:
        idx = np.random.randint(0, episode_len(episode) - traj_length + 1) + 1
    Si episode_len(episode) < traj_length, randint recibe high <= 0 y arroja
    ValueError -- no hay padding/masking para episodios mas cortos que la
    ventana. Por eso este script FILTRA en la conversion: solo escribe
    episodios con T >= --min_length, y reporta cuantos/que fraccion se
    descartaron -- para que la decision quede visible, no oculta.
    --min_length DEBE ser >= traj_length de la config de entrenamiento que
    se vaya a usar (dt.yaml/hdt.yaml actual: traj_length=20).
"""
import argparse
import io
from pathlib import Path

import numpy as np


def convert_episode(npy_path: Path) -> dict:
    raw = np.load(npy_path, allow_pickle=True).item()

    obs_chw = np.asarray(raw["observations"])  # (T+1, 3, 64, 64) uint8
    actions = np.asarray(raw["actions"])  # (T, 1) int64
    rewards = np.asarray(raw["rewards"])  # (T, 1) float32
    dones = np.asarray(raw["dones"])  # (T,)   float32

    T = actions.shape[0]
    assert obs_chw.shape[0] == T + 1, (
        f"{npy_path.name}: esperaba T+1={T + 1} observaciones, "
        f"encontre {obs_chw.shape[0]}"
    )

    # CHW -> HWC (convencion de obs de pixeles del resto del repo)
    obs_hwc = obs_chw.transpose(0, 2, 3, 1)  # (T+1, 64, 64, 3)
    if obs_hwc.dtype != np.uint8:
        obs_hwc = np.round(obs_hwc).astype(np.uint8)

    # Dummy en idx 0 para igualar la convencion T+1 que usan episode_len()/_sample()
    action_pad = np.concatenate([np.zeros_like(actions[:1]), actions], axis=0)  # (T+1, 1)
    reward_pad = np.concatenate([np.zeros_like(rewards[:1]), rewards], axis=0)  # (T+1, 1)
    done_pad = np.concatenate([np.zeros_like(dones[:1]), dones], axis=0)  # (T+1,)
    discount = np.ones_like(reward_pad)  # (T+1, 1) -- sintetico, BC no lo usa en el loss

    return {
        "observation": obs_hwc,
        "action": action_pad,
        "reward": reward_pad,
        "discount": discount,
        "done": done_pad,
    }


def save_episode(episode: dict, out_path: Path):
    """Idem a save_episode() de replay_buffer.py -- mismo formato, sin importar
    el modulo para no acoplar este script a estar dentro del repo."""
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with out_path.open("wb") as f:
            f.write(bs.read())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src_dir", required=True, help="carpeta con los .npy crudos (ej. raw_data/coinrun/data)")
    ap.add_argument("--out_dir", required=True, help="carpeta destino para los .npz convertidos")
    ap.add_argument(
        "--min_length",
        type=int,
        default=20,
        help="largo minimo de episodio (T = num. acciones) para conservarlo. "
        "DEBE ser >= traj_length del config que se vaya a usar para entrenar "
        "(dt.yaml/hdt.yaml actual: traj_length=20).",
    )
    args = ap.parse_args()

    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(src_dir.rglob("*.npy"))
    print(f"Encontrados {len(files)} episodios crudos en {src_dir}")
    if not files:
        raise SystemExit(f"No se encontraron .npy en {src_dir} -- revisa la ruta.")

    n_kept, n_dropped = 0, 0
    dropped_lengths = []

    for i, f in enumerate(files):
        ep = convert_episode(f)
        T = ep["action"].shape[0] - 1  # -1 por el dummy en idx 0
        if T < args.min_length:
            n_dropped += 1
            dropped_lengths.append(T)
            continue
        out_path = out_dir / f"{f.stem}.npz"
        save_episode(ep, out_path)
        n_kept += 1

        if (i + 1) % 2000 == 0:
            print(f"  ... procesados {i + 1}/{len(files)}", flush=True)

    print(f"\nConvertidos y guardados: {n_kept}")
    print(
        f"Descartados (T < {args.min_length}): {n_dropped} "
        f"({100 * n_dropped / len(files):.1f}% del total)"
    )
    if dropped_lengths:
        dl = np.array(dropped_lengths)
        print(f"  largos descartados: min={dl.min()}, max={dl.max()}, media={dl.mean():.1f}")
    print(f"\nSalida en: {out_dir}")


if __name__ == "__main__":
    main()
