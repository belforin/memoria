"""
Control de METODOLOGIA_DT_HDT.md seccion 7.4: corre pretrain_coinrun.py con
el encoder de pixeles intacto, pero sumando a su salida ruido gaussiano de
desviacion E2E_NOISE_STD (variable de entorno; ~1e-7 = ultimo bit de
float32 para valores de orden 1). Sirve para medir cuanto amplifica el
entrenamiento una perturbacion del tamano del redondeo, y compararlo con la
diferencia pixeles vs. embeddings precalculados. No se usa para entrenar.
"""
import os
import sys

import torch

from agent.modules.pixel_encoder import ImpalaProcgenEncoder

std = float(os.environ.get("E2E_NOISE_STD", "0"))
gen = torch.Generator(device="cuda")
gen.manual_seed(int(os.environ.get("E2E_NOISE_SEED", "0")))
_forward = ImpalaProcgenEncoder.forward


def noisy_forward(self, obs):
    out = _forward(self, obs)
    if std > 0:
        out = out + std * torch.randn(out.shape, device=out.device, generator=gen)
    return out


ImpalaProcgenEncoder.forward = noisy_forward

if __name__ == "__main__":
    import runpy

    print(f"[e2e_noise] E2E_NOISE_STD={std}", flush=True)
    # run_path como __main__ (no import): Hydra resuelve config_path="."
    # relativo al archivo solo cuando el script se ejecuta directamente
    runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrain_coinrun.py"),
                   run_name="__main__")
