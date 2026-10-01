"""
Test standalone de agent/bc_ar.py (BC autoregresivo sin return-to-go,
belforin/memoria METODOLOGIA_DT_HDT.md secciones 7.5-7.6). Tensores dummy en
CPU, salvo la carga del IMPALA que lee el checkpoint preentrenado.
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from agent.bc_ar import BCARAgent, BCAREvalAgent, BCARModel

IMPALA_CKPT = "/home/bmancilla/archive/MaskDP/pretrained_encoders/procgen_coinrun_easy_encoder.pt"
torch.manual_seed(0)
failures = []


def check(name, cond):
    print(f"[{'OK' if cond else 'FAIL'}] {name}")
    if not cond:
        failures.append(name)


def small_cfg(topology, discrete=False, **kw):
    c = dict(topology=topology, n_embd=32, n_head=2, n_layer=2, n_obs_layer=1, n_act_layer=1,
             traj_length=12, embd_pdrop=0.0, attn_pdrop=0.0, resid_pdrop=0.0, mlp_pdrop=0.0,
             weight_decay=0.0, use_pixel_obs=False, discrete_actions=discrete, num_actions=15)
    c.update(kw)
    return SimpleNamespace(**c)


def actions(B, T, discrete, A=6):
    if discrete:
        return torch.randint(0, 15, (B, T, 1))
    return torch.rand(B, T, A) * 2 - 1


B, T, OBS, A = 4, 12, 17, 6

# 1. forward, causalidad, loss enmascarada
for topology in ["uni", "hier"]:
    for discrete in [False, True]:
        tag = f"{topology}{' discreto' if discrete else ''}"
        model = BCARModel(OBS, 15 if discrete else A, small_cfg(topology, discrete)).eval()
        out_dim = 15 if discrete else A
        obs, act = torch.randn(B, T, OBS), actions(B, T, discrete)
        pred = model(obs, act)
        check(f"[{tag}] shape {tuple(pred.shape)}", pred.shape == (B, T, out_dim))

        k = 6
        act_b = act.clone()
        act_b[:, k:] = (act_b[:, k:] + 1) % 15 if discrete else -act_b[:, k:]
        pred_b = model(obs, act_b)
        check(f"[{tag}] causal: a_k.. no cambian pred[:k+1]", torch.allclose(pred[:, : k + 1], pred_b[:, : k + 1], atol=1e-6))
        check(f"[{tag}] causal: a_k cambia pred[k+1:]", not torch.allclose(pred[:, k + 1 :], pred_b[:, k + 1 :], atol=1e-6))
        obs_c = obs.clone()
        obs_c[:, k] += 1.0
        pred_c = model(obs_c, act)
        check(f"[{tag}] causal: s_k no cambia pred[:k], si pred[k]",
              torch.allclose(pred[:, :k], pred_c[:, :k], atol=1e-6) and not torch.allclose(pred[:, k], pred_c[:, k], atol=1e-6))

        # historial incompleto (evaluacion): menos acciones que estados
        pred_short = model(obs[:, :k + 1], act[:, :k])
        check(f"[{tag}] T acciones-1 = misma pred que con a_k", torch.allclose(pred_short[:, -1], pred[:, k], atol=1e-5))

        mask = torch.ones(B, T, 1)
        mask[:, 8:] = 0
        act_pad = actions(B, T, discrete)
        act_pad[:, :8] = act[:, :8]
        l1 = model.action_loss(model(obs, act), act, mask)
        l2 = model.action_loss(model(obs, act_pad), act_pad, mask)
        check(f"[{tag}] mascara: relleno no cambia la loss", abs(l1.item() - l2.item()) < 1e-5)

# 2. agente de entrenamiento con el formato de batch de replay_buffer.py
for topology in ["uni", "hier"]:
    for discrete in [False, True]:
        tag = f"{topology}{' discreto' if discrete else ''}"
        agent = BCARAgent("bc", (OBS,), (A,), "cpu", 1e-3, B, True, small_cfg(topology, discrete))
        obs, act = torch.randn(B, T, OBS), actions(B, T, discrete).float()
        batch = (obs.numpy(), act.numpy(), np.zeros((B, T, 1), np.float32), np.ones((B, T, 1), np.float32),
                 obs.numpy(), np.ones((B, T, 1), np.float32))

        def it():
            while True:
                yield batch

        losses = [agent.update(it(), s)["action_loss"] for s in range(60)]
        check(f"[{tag}] update: loss baja {losses[0]:.3f} -> {losses[-1]:.3f}", losses[-1] < 0.5 * losses[0])

# 3. agente de evaluacion: interfaz de eval_return.py / eval_bct.py
for topology in ["uni", "hier"]:
    for discrete in [False, True]:
        tag = f"{topology}{' discreto' if discrete else ''}"
        cfg = small_cfg(topology, discrete)
        ev = BCAREvalAgent((OBS,), (A,), "cpu", K=None, sample=discrete, transformer_cfg=cfg)
        ev.reset()
        outs = [ev.act(np.random.randn(OBS).astype(np.float32)) for _ in range(T + 5)]
        if discrete:
            ok = all(o.shape == (1,) and o.dtype == np.int64 and 0 <= o[0] < 15 for o in outs)
        else:
            ok = all(o.shape == (A,) and np.all(np.abs(o) <= 1) for o in outs)
        check(f"[{tag}] eval: {len(outs)} pasos (> K={ev.K}), salida valida", ok and len(ev._obs_buffer) == ev.K)

        # la accion del agente de eval == prediccion del modelo sobre la misma ventana
        ev2 = BCAREvalAgent((OBS,), (A,), "cpu", K=None, sample=False, transformer_cfg=cfg)
        ev2.mdp.load_state_dict(ev.mdp.state_dict())
        ev2.reset()
        obs_seq = [np.random.randn(OBS).astype(np.float32) for _ in range(5)]
        acts = [ev2.act(o) for o in obs_seq]
        with torch.no_grad():
            ref = ev2.mdp(torch.as_tensor(np.stack(obs_seq))[None],
                          torch.as_tensor(np.stack(acts[:-1]))[None])[0, -1]
        last = acts[-1]
        same = int(torch.argmax(ref)) == int(last[0]) if discrete else np.allclose(ref.numpy(), last, atol=1e-5)
        check(f"[{tag}] eval: accion = modelo sobre la ventana", same)

# 4. IMPALA: convoluciones del checkpoint, proyeccion aleatoria, todo congelado
if os.path.exists(IMPALA_CKPT):
    ckpt = torch.load(IMPALA_CKPT, map_location="cpu")
    cfg = OmegaConf.load(os.path.join(ROOT, "agent", "bc_uni_procgen.yaml")).transformer_cfg
    cfg.pretrained_encoder_path = IMPALA_CKPT
    m = BCARModel(None, 15, cfg)
    enc = m.state_embed
    conv_same = all(torch.equal(v, ckpt["convnet"][k]) for k, v in enc.convnet.state_dict().items())
    proj_diff = not torch.equal(enc.projection.weight, ckpt["projection"]["weight"])
    frozen = all(not p.requires_grad for p in enc.parameters())
    check(f"IMPALA: conv del checkpoint {conv_same}, proyeccion aleatoria {proj_diff}, congelado {frozen}",
          conv_same and proj_diff and frozen)
    pred = m(torch.randint(0, 255, (2, 16, 64, 64, 3), dtype=torch.uint8), torch.randint(0, 15, (2, 16, 1)))
    check(f"IMPALA: forward con pixeles {tuple(pred.shape)}", pred.shape == (2, 16, 15))
else:
    print(f"[SKIP] {IMPALA_CKPT} no existe")

# 5. parametros entrenables vs. Tabla 3.1 de la tesis / sus modelos (seccion 7.5-7.6)
ref = {"bc_uni": {17: 4094487, 24: 4098078, 78: 4128858},
       "bc_hier": {17: 4161815, 24: 4164510, 78: 4187610}}
for name in ["bc_uni", "bc_hier"]:
    cfg = OmegaConf.load(os.path.join(ROOT, "agent", f"{name}.yaml")).transformer_cfg
    for obs_dim, act_dim in [(17, 6), (24, 6), (78, 12)]:
        m = BCARModel(obs_dim, act_dim, cfg)
        n = sum(p.numel() for p in m.parameters() if p.requires_grad)
        dev = 100 * (n - ref[name][obs_dim]) / ref[name][obs_dim]
        check(f"{name} DMC obs={obs_dim}: {n:,} params ({dev:+.1f}% vs Benjamin)", abs(dev) < 6)

print()
if failures:
    print(f"{len(failures)} FALLAS: {failures}")
    sys.exit(1)
print("todo OK")
