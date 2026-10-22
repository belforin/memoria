"""
Test de analysis/input_attribution.py (metrica 4, METODOLOGIA_DT_HDT.md
seccion 7.16) con modelos chicos y episodios sinteticos en CPU.
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
from analysis.input_attribution import aopc, by_lag, history_only, integrated_gradients, spearman_rows

torch.manual_seed(0)
failures = []


def check(name, cond):
    print(f"[{'OK' if cond else 'FAIL'}] {name}")
    if not cond:
        failures.append(name)


def cfg(topology, discrete=False, **kw):
    c = dict(topology=topology, n_embd=32, n_head=2, n_layer=2, n_obs_layer=1, n_act_layer=1,
             traj_length=6, embd_pdrop=0.0, attn_pdrop=0.0, resid_pdrop=0.0, mlp_pdrop=0.0,
             weight_decay=0.0, use_pixel_obs=False, discrete_actions=discrete, num_actions=5)
    c.update(kw)
    return OmegaConf.create(c)


OBS, A, K, N = 5, 3, 6, 8

for topology, kw in [("uni", {}), ("hier", {}), ("hier", dict(enc_n_embd=16, n_layer=3))]:
    for discrete in [False, True]:
        tag = f"{topology}{'_enc' if kw else ''}{' discreto' if discrete else ''}"
        m = BCARModel(OBS, 5 if discrete else A, cfg(topology, discrete, **kw)).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        obs = torch.randn(N, K, OBS)
        act = torch.randint(0, 5, (N, K - 1, 1)) if discrete else torch.rand(N, K - 1, A) * 2 - 1
        with torch.no_grad():
            s, a = m.embed_tokens(obs, act)
        check(f"[{tag}] forward_tokens(embed_tokens) = forward",
              torch.allclose(m.forward_tokens(s, a), m(obs, act), atol=1e-6))

        sb, ab = s.mean((0, 1)), a[:, :K - 1].mean((0, 1))
        # completitud: sum attr -> F(x) - F(base) al crecer m (converge ~1/m)
        e64 = integrated_gradients(m, s, a, sb, ab, m=64, chunk=4)[2].median()
        e512 = integrated_gradients(m, s, a, sb, ab, m=512, chunk=4)[2].median()
        check(f"[{tag}] IG completitud: err rel. mediano m=64 {e64:.2e} -> m=512 {e512:.2e}", e512 < e64 and e512 < 0.03)

        # un token igual a la linea base no recibe atribucion
        s2 = s.clone()
        s2[:, 2] = sb
        at_s, at_a, _ = integrated_gradients(m, s2, a, sb, ab, m=16, chunk=4)
        check(f"[{tag}] token = linea base -> atribucion 0", torch.allclose(at_s[:, 2], torch.zeros(N), atol=1e-6)
              and at_s.shape == (N, K) and at_a.shape == (N, K - 1))

        # por retraso: s en la posicion K-1 es k=0; a en la posicion K-2 es k=1
        sl, al = by_lag(at_s, at_a)
        check(f"[{tag}] by_lag", torch.equal(sl[:, 0], at_s[:, K - 1]) and torch.equal(al[:, 1], at_a[:, K - 2])
              and torch.isnan(al[:, 0]).all())

        # AOPC: k tokens reemplazados por si mismos -> 0; mas tokens -> efecto >= 0
        pool_o = obs[0].clone()
        pool_a = act[0].clone()
        sc = torch.rand(N, 2 * K - 1)
        same = aopc(m, obs[:1].repeat(N, 1, 1), act[:1].repeat(N, *[1] * (act.dim() - 1)),
                    pool_o[K - 1:K].repeat(3, 1), pool_a[:1].repeat(3, *[1] * (act.dim() - 2)),
                    torch.tensor([[0.0] * (K - 1) + [1.0] + [0.0] * (K - 1)]).repeat(N, 1), [1], 2,
                    np.random.default_rng(0))
        check(f"[{tag}] AOPC: reemplazar s_t por el mismo s_t -> 0", abs(same[0]) < 1e-6)

        # AOPC historia: historia constante e igual a los donantes, s_t distinto -> 0 aunque s_t sea el top
        o_c = obs[:1, :1].repeat(N, K, 1)
        o_c[:, K - 1] = obs[1, K - 1]
        a_c = act[:1, :1].repeat(N, K - 1, *[1] * (act.dim() - 2))
        top_st = torch.tensor([[0.0] * (K - 1) + [1.0] + [0.5] * (K - 1)]).repeat(N, 1)
        args_c = (o_c[0, :1].repeat(3, 1), a_c[0, :1].repeat(3, *[1] * (act.dim() - 2)))
        full = aopc(m, o_c, a_c, *args_c, top_st, [1, 2], 2, np.random.default_rng(0))
        hist = aopc(m, o_c, a_c, *args_c, history_only(top_st), [1, 2], 2, np.random.default_rng(0))
        check(f"[{tag}] AOPC historia no reemplaza s_t: {hist} vs. completo {full}",
              np.abs(hist).max() < 1e-6 and full.min() > 1e-4)

check("Spearman por fila: identico = 1, invertido = -1",
      torch.allclose(spearman_rows(torch.arange(5.)[None], torch.arange(5.)[None]), torch.ones(1))
      and torch.allclose(spearman_rows(torch.arange(5.)[None], -torch.arange(5.)[None]), -torch.ones(1)))

# de punta a punta
tmp = tempfile.mkdtemp()
d = os.path.join(tmp, "eps")
os.makedirs(d)
rng = np.random.default_rng(1)
for e in range(40):
    L = int(rng.integers(K + 2, 30))
    act = rng.uniform(-1, 1, (L + 1, A)).astype(np.float32)
    act[0] = 0
    np.savez(os.path.join(d, f"ep{e:02d}.npz"), observation=rng.normal(size=(L + 1, OBS)).astype(np.float32),
             action=act)
c = cfg("hier", enc_n_embd=16, n_layer=3)
snap = os.path.join(tmp, "snap.pt")
torch.save({"cfg": c, "model": BCARModel(OBS, A, c).state_dict()}, snap)
out = os.path.join(tmp, "res", "x")
r = subprocess.run([sys.executable, os.path.join(ROOT, "analysis", "input_attribution.py"), "--snapshot", snap,
                    "--data_dir", d, "--out", out, "--n_windows", "16", "--n_pool", "64", "--m", "8",
                    "--device", "cpu"], capture_output=True, text=True)
ok = r.returncode == 0 and os.path.exists(out + ".json")
check("script de punta a punta" + ("" if ok else f": {r.stderr[-800:]}"), ok)
if ok:
    s = json.load(open(out + ".json"))
    check(f"resumen: AOPC {s['aopc']}, ks {s['aopc_ks']}", set(s["aopc"]) == {"ig", "occlusion", "recency", "random"}
          and s["aopc_ks"] == [1, 2, 4, 8] and set(s["aopc_hist"]) == set(s["aopc"]))

print()
if failures:
    print(f"{len(failures)} FALLAS: {failures}")
    sys.exit(1)
print("todo OK")
