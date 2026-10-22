"""
Test de analysis/lag_ablation.py (metrica 2, METODOLOGIA_DT_HDT.md seccion
7.16) con modelos chicos y episodios sinteticos en CPU.
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from agent.bc_ar import BCARModel
from analysis.lag_ablation import distance, lag_profile, predict_last, sample_windows, to_inputs

torch.manual_seed(0)
failures = []


def check(name, cond):
    print(f"[{'OK' if cond else 'FAIL'}] {name}")
    if not cond:
        failures.append(name)


def cfg(topology, discrete=False, **kw):
    c = dict(topology=topology, n_embd=32, n_head=2, n_layer=2, n_obs_layer=1, n_act_layer=1,
             traj_length=8, embd_pdrop=0.0, attn_pdrop=0.0, resid_pdrop=0.0, mlp_pdrop=0.0,
             weight_decay=0.0, use_pixel_obs=False, discrete_actions=discrete, num_actions=5)
    c.update(kw)
    return OmegaConf.create(c)


OBS, A, K = 5, 3, 8
tmp = tempfile.mkdtemp()

# 1. alineacion de ventanas: obs[j] = j, action[j] = j (action[0] dummy)
d1 = os.path.join(tmp, "align")
os.makedirs(d1)
for e in range(12):
    L = 20
    obs = np.tile(np.arange(L + 1, dtype=np.float32)[:, None], (1, OBS)) + 100 * e
    act = np.tile(np.arange(L + 1, dtype=np.float32)[:, None], (1, A)) + 100 * e
    np.savez(os.path.join(d1, f"ep{e:02d}.npz"), observation=obs, action=act)
files = sorted(os.listdir(d1))
o, a, tgt, prev, po, pa = sample_windows([os.path.join(d1, f) for f in files], K, 6, 16, np.random.default_rng(0))
ok = all(np.array_equal(a[n, :, 0], o[n, :K - 1, 0] + 1) for n in range(6))
check("ventana: a_j acompana a s_j (action[i+1+j] con obs[i+j])", ok)
check("ventana: a_t = s_t + 1 y a_{t-1} = s_t", np.array_equal(tgt[:, 0], o[:, -1, 0] + 1) and np.array_equal(prev[:, 0], o[:, -1, 0]))
win_eps = set((o[:, 0, 0] // 100).astype(int))
pool_eps = set((po[:, 0] // 100).astype(int))
check(f"donantes de otros episodios (ventanas {sorted(win_eps)}, pool {sorted(pool_eps)})", not (win_eps & pool_eps))

for topology, kw in [("uni", {}), ("hier", {}), ("hier", dict(enc_n_embd=16, n_layer=3))]:
    for discrete in [False, True]:
        tag = f"{topology}{'_enc' if kw else ''}{' discreto' if discrete else ''}"
        m = BCARModel(OBS, 5 if discrete else A, cfg(topology, discrete, **kw)).eval()
        N = 10
        obs = torch.randn(N, K, OBS)
        act = torch.randint(0, 5, (N, K - 1, 1)) if discrete else torch.rand(N, K - 1, A) * 2 - 1

        # 2. donante identico al token original -> distancia 0 en todo retraso
        same = lag_profile(m, obs[:1].repeat(N, 1, 1), act[:1].repeat(N, *[1] * (act.dim() - 1)),
                           obs[0, :1].repeat(4, 1), act[0, :1].repeat(4, *[1] * (act.dim() - 2)), 2,
                           np.random.default_rng(0))
        # con pool = un unico token, solo el reemplazo en la posicion de ese token da 0
        z_s = same["d_state"][K - 1]          # s en la posicion 0 = obs[0, 0]
        z_a = same["d_act"][K - 1]            # a en la posicion 0 = act[0, 0]
        check(f"[{tag}] donante identico: d = 0", np.allclose(z_s, 0, atol=1e-6) and np.allclose(z_a, 0, atol=1e-6))

        # 3. coincide con un calculo manual de un retraso
        pool_o, pool_a = torch.randn(32, OBS), (torch.randint(0, 5, (32, 1)) if discrete else torch.rand(32, A))
        prof = lag_profile(m, obs, act, pool_o, pool_a, 3, np.random.default_rng(5))
        rng = np.random.default_rng(5)
        base = predict_last(m, obs, act).repeat(3, 1)
        o_r, a_r = obs.repeat(3, 1, 1), act.repeat(3, *[1] * (act.dim() - 1))
        ref = {}
        for k in range(K):
            oo = o_r.clone()
            oo[:, K - 1 - k] = pool_o[torch.as_tensor(rng.integers(0, 32, size=3 * N))]
            ref[("s", k)] = distance(m, base, predict_last(m, oo, a_r))[0].reshape(3, N).mean(0)
            if k >= 1:
                aa = a_r.clone()
                aa[:, K - 1 - k] = pool_a[torch.as_tensor(rng.integers(0, 32, size=3 * N))]
                ref[("a", k)] = distance(m, base, predict_last(m, o_r, aa))[0].reshape(3, N).mean(0)
        ok = all(np.allclose(prof["d_state"][k], ref[("s", k)].numpy(), atol=1e-6) for k in range(K))
        ok &= all(np.allclose(prof["d_act"][k], ref[("a", k)].numpy(), atol=1e-6) for k in range(1, K))
        check(f"[{tag}] lag_profile = calculo manual por retraso", ok)
        check(f"[{tag}] efectos > 0 en estados y acciones pasadas",
              np.nanmean(prof["d_state"][0]) > 0 and np.nanmean(prof["d_act"][1]) > 0
              and np.isnan(prof["d_act"][0]).all())

# 4. de punta a punta: snapshot chico + episodios sinteticos
d2 = os.path.join(tmp, "eps")
os.makedirs(d2)
rng = np.random.default_rng(1)
for e in range(40):
    L = int(rng.integers(K + 2, 30))
    act = rng.uniform(-1, 1, (L + 1, A)).astype(np.float32)
    act[0] = 0
    np.savez(os.path.join(d2, f"ep{e:02d}.npz"), observation=rng.normal(size=(L + 1, OBS)).astype(np.float32), action=act)
c = cfg("hier", enc_n_embd=16, n_layer=3)
m = BCARModel(OBS, A, c)
snap = os.path.join(tmp, "snap.pt")
torch.save({"cfg": c, "model": m.state_dict()}, snap)
out = os.path.join(tmp, "res", "x")
r = subprocess.run([sys.executable, os.path.join(ROOT, "analysis", "lag_ablation.py"), "--snapshot", snap,
                    "--data_dir", d2, "--out", out, "--n_windows", "16", "--n_pool", "64", "--device", "cpu"],
                   capture_output=True, text=True)
ok = r.returncode == 0 and os.path.exists(out + ".json") and os.path.exists(out + ".npz")
check("script de punta a punta" + ("" if ok else f": {r.stderr[-500:]}"), ok)
if ok:
    s = json.load(open(out + ".json"))
    check(f"resumen: {len(s['state_curve'])} retrasos, frac_action_past={s['frac_action_past']:.3f}, "
          f"lag_90={s['lag_90']}", len(s["state_curve"]) == K and 0 <= s["frac_action_past"] <= 1)

print()
if failures:
    print(f"{len(failures)} FALLAS: {failures}")
    sys.exit(1)
print("todo OK")
