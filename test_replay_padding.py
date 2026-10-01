"""
Test standalone de OfflineReplayBuffer con pad_short_episodes=True (Etapa BC,
METODOLOGIA_DT_HDT.md seccion 7) sobre episodios reales de data/coinrun:
uno mas corto que traj_length=64 se rellena con ceros y se enmascara, uno
mas largo se muestrea igual que antes con mascara de unos.
"""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(ROOT))

import numpy as np

from replay_buffer import OfflineReplayBuffer

TRAJ_LENGTH = 64
EPISODES = ["episode_0_20.npz", "episode_10000_38.npz", "episode_10001_70.npz"]

failures = []


def check(name, cond):
    status = "OK" if cond else "FAIL"
    print(f"[{status}] {name}")
    if not cond:
        failures.append(name)


with tempfile.TemporaryDirectory() as tmp:
    for fn in EPISODES:
        os.symlink(ROOT / "data" / "coinrun" / fn, Path(tmp) / fn)

    rb = OfflineReplayBuffer(
        None, Path(tmp), 10**8, 1, 0.99, "coinrun", TRAJ_LENGTH, None, None,
        False, "pixels", return_to_go=False, pad_short_episodes=True,
    )
    rb._load(False)
    rb._loaded = True

    for fn in rb._episode_fns:
        ep = rb._episodes[fn]
        L = next(iter(ep.values())).shape[0] - 1
        rb._sample_episode = lambda ep=ep: ep
        out = rb._sample()
        check(f"{fn.name} (L={L}): 7 elementos", len(out) == 7)
        obs, act, rew, disc, nobs, ts, mask = out
        check(f"{fn.name}: obs {obs.shape}", obs.shape == (TRAJ_LENGTH, 64, 64, 3))
        if L < TRAJ_LENGTH:
            check(f"{fn.name}: mascara = {L} pasos reales", int(mask.sum()) == L and mask[:L].all())
            check(f"{fn.name}: obs/accion reales intactas",
                  np.array_equal(obs[:L], ep["observation"][:L])
                  and np.array_equal(act[:L], ep["action"][1 : L + 1]))
            check(f"{fn.name}: relleno en cero", (obs[L:] == 0).all() and (act[L:] == 0).all())
            check(f"{fn.name}: timesteps 0..T-1", ts[0, 0] == 0 and ts[-1, 0] == TRAJ_LENGTH - 1)
        else:
            check(f"{fn.name}: mascara de unos", int(mask.sum()) == TRAJ_LENGTH)

print()
if failures:
    print(f"{len(failures)} FALLAS: {failures}")
    sys.exit(1)
print("todo OK")
