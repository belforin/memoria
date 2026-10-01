"""
Precalcula los embeddings del encoder IMPALA congelado sobre todos los
frames del dataset CoinRun (Etapa BC, METODOLOGIA_DT_HDT.md seccion 7.4).

El encoder (agent/modules/pixel_encoder.py::ImpalaProcgenEncoder, cargado
con load_procgen_impala y congelado) solo tiene Conv/MaxPool/ReLU/Linear,
sin BatchNorm ni Dropout, y no se entrena: su salida es una funcion fija de
cada frame. Recalcularla en cada paso de entrenamiento (128 x 64 = 8192
frames por batch) dominaba el tiempo del smoke test (job 30546). Con los
embeddings guardados, DecisionTransformer/HierarchicalDecisionTransformer
reciben obs (B, T, 256) y se saltan el encoder (ver agent/dt.py::forward).

Escribe en --dst un .npz por episodio con el mismo nombre y las mismas
claves que --src, salvo `observation`: (T+1, 64, 64, 3) uint8 pasa a
(T+1, feature_dim) float32. FP32 estricto (TF32 desactivado: las GPUs
Ampere como las A40 lo usan por defecto en convoluciones).

Uso: python precompute_coinrun_embeddings.py --src data/coinrun --dst data_emb/coinrun
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from agent.modules.load_pretrained_encoder import load_procgen_impala
from agent.modules.pixel_encoder import PixelEncoder

OBS_SHAPE = (64, 64, 3)


def build_encoder(ckpt, feature_dim, device):
    encoder = PixelEncoder(OBS_SHAPE, feature_dim, encoder_type="procgen_impala")
    load_procgen_impala(encoder, ckpt, freeze=True)
    return encoder.to(device).eval()


@torch.no_grad()
def embed_frames(encoder, frames, device, chunk):
    """frames: (N, H, W, C) uint8 -> (N, feature_dim) float32"""
    out = []
    for i in range(0, len(frames), chunk):
        x = torch.as_tensor(frames[i : i + chunk], device=device)
        out.append(encoder(x).float().cpu().numpy())
    return np.concatenate(out, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/coinrun")
    parser.add_argument("--dst", default="data_emb/coinrun")
    parser.add_argument("--ckpt", default="pretrained_encoders/procgen_coinrun_easy_encoder.pt")
    parser.add_argument("--feature-dim", type=int, default=256,
                         help="Debe ser n_embd del modelo (256, igual al checkpoint)")
    parser.add_argument("--chunk", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    encoder = build_encoder(args.ckpt, args.feature_dim, args.device)

    files = sorted(src.glob("episode_*.npz"))
    print(f"{len(files)} episodios en {src} -> {dst}")
    t0 = time.time()
    n_frames = 0
    for k, fn in enumerate(files):
        out_fn = dst / fn.name
        if out_fn.exists():
            continue
        with fn.open("rb") as f:
            episode = dict(np.load(f))
        frames = episode["observation"]
        assert frames.dtype == np.uint8 and frames.shape[1:] == OBS_SHAPE, (fn, frames.shape)
        episode["observation"] = embed_frames(encoder, frames, args.device, args.chunk)
        n_frames += len(frames)
        # escribir a un temporal y renombrar: un job cortado no deja
        # archivos a medias que el skip de arriba daria por buenos
        tmp = dst / (fn.name + ".part")
        with tmp.open("wb") as f:
            np.savez(f, **episode)
        tmp.rename(out_fn)
        if (k + 1) % 1000 == 0:
            print(f"  {k + 1}/{len(files)} episodios, {n_frames} frames, "
                  f"{time.time() - t0:.0f}s", flush=True)
    print(f"listo: {n_frames} frames nuevos en {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
