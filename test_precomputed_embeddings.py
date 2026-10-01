"""
Verifica los embeddings precalculados de CoinRun (precompute_coinrun_embeddings.py,
METODOLOGIA_DT_HDT.md seccion 7.4) contra el camino con pixeles:

1. el embedding guardado en data_emb/coinrun coincide con recalcular el
   encoder sobre los frames de data/coinrun;
2. BC-uni y BC-hier (configs reales agent/bc_{uni,hier}_coinrun.yaml, pesos
   aleatorios + IMPALA preentrenado) dan la misma prediccion con pixeles que
   con embeddings, sobre ventanas reales con relleno.

Necesita GPU si se corre con --device cuda (igual que el entrenamiento).
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from omegaconf import OmegaConf

from agent.dt import DecisionTransformer
from agent.hdt import HierarchicalDecisionTransformer
from precompute_coinrun_embeddings import build_encoder, embed_frames

EPISODES = ["episode_0_20.npz", "episode_10000_38.npz", "episode_10001_70.npz"]

failures = []


def check(name, cond):
    status = "OK" if cond else "FAIL"
    print(f"[{status}] {name}")
    if not cond:
        failures.append(name)


def window(ep, T):
    """primeros min(L, T) pasos, rellenados hasta T como _sample_padded"""
    L = min(len(ep["action"]) - 1, T)
    pad = T - L

    def padded(x):
        return np.concatenate([x, np.zeros((pad,) + x.shape[1:], dtype=x.dtype)], axis=0)

    return padded(ep["observation"][:L]), padded(ep["action"][1 : L + 1]), L


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pix-dir", default="data/coinrun")
    parser.add_argument("--emb-dir", default="data_emb/coinrun")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = args.device
    ckpt = str(ROOT / "pretrained_encoders/procgen_coinrun_easy_encoder.pt")

    encoder = build_encoder(ckpt, 256, device)
    pix = {fn: dict(np.load(Path(args.pix_dir) / fn)) for fn in EPISODES}
    emb = {fn: dict(np.load(Path(args.emb_dir) / fn)) for fn in EPISODES}

    # 1. embedding guardado == recalculado; resto de las claves identicas
    for fn in EPISODES:
        fresh = embed_frames(encoder, pix[fn]["observation"], device, 4096)
        stored = emb[fn]["observation"]
        err = np.abs(fresh - stored).max()
        check(f"{fn}: embedding guardado {stored.shape} == recalculado (max |diff| {err:.2e})",
              stored.shape == (len(pix[fn]["observation"]), 256) and err < 1e-5)
        same_keys = all(np.array_equal(pix[fn][k], emb[fn][k]) for k in pix[fn] if k != "observation")
        check(f"{fn}: resto de las claves identicas", same_keys and set(pix[fn]) == set(emb[fn]))

    # 2. modelo con pixeles == modelo con embeddings
    for yaml, Model in [("bc_uni_coinrun", DecisionTransformer), ("bc_hier_coinrun", HierarchicalDecisionTransformer)]:
        cfg = OmegaConf.load(ROOT / "agent" / f"{yaml}.yaml").transformer_cfg
        cfg.pretrained_encoder_path = ckpt
        T = cfg.traj_length
        torch.manual_seed(0)
        model = Model((64, 64, 3), cfg.num_actions, cfg).to(device).eval()
        for fn in EPISODES:
            obs_p, act, L = window(pix[fn], T)
            obs_e, _, _ = window(emb[fn], T)
            a = torch.as_tensor(act, device=device)[None]
            with torch.no_grad():
                out_p = model(None, torch.as_tensor(obs_p, device=device)[None], a)
                out_e = model(None, torch.as_tensor(obs_e, device=device)[None], a)
            err = (out_p - out_e)[:, :L].abs().max().item()
            same_argmax = bool((out_p[:, :L].argmax(-1) == out_e[:, :L].argmax(-1)).all())
            check(f"{yaml} {fn}: pixeles == embeddings (max |diff| logits {err:.2e}, argmax igual)",
                  err < 1e-4 and same_argmax)

    print()
    if failures:
        print(f"{len(failures)} FALLAS: {failures}")
        sys.exit(1)
    print("todo OK")


if __name__ == "__main__":
    main()
