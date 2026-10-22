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


def small_cfg(topology="uni", discrete=False, **kw):
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
# (topologia, overrides de small_cfg); hier_enc = stacks por modalidad mas
# angostos que el conjunto (enc_n_embd, seccion 7.15)
VARIANTS = [("uni", {}), ("hier", {}), ("hier_enc", dict(topology="hier", enc_n_embd=16, n_layer=3))]

# 1. forward, causalidad, loss enmascarada
for topology, kw in VARIANTS:
    for discrete in [False, True]:
        tag = f"{topology}{' discreto' if discrete else ''}"
        kw = dict(dict(topology=topology), **kw)
        model = BCARModel(OBS, 15 if discrete else A, small_cfg(discrete=discrete, **kw)).eval()
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
for topology, kw in VARIANTS:
    for discrete in [False, True]:
        tag = f"{topology}{' discreto' if discrete else ''}"
        kw = dict(dict(topology=topology), **kw)
        agent = BCARAgent("bc", (OBS,), (A,), "cpu", 1e-3, B, True, small_cfg(discrete=discrete, **kw))
        obs, act = torch.randn(B, T, OBS), actions(B, T, discrete).float()
        batch = (obs.numpy(), act.numpy(), np.zeros((B, T, 1), np.float32), np.ones((B, T, 1), np.float32),
                 obs.numpy(), np.ones((B, T, 1), np.float32))

        def it():
            while True:
                yield batch

        losses = [agent.update(it(), s)["action_loss"] for s in range(60)]
        check(f"[{tag}] update: loss baja {losses[0]:.3f} -> {losses[-1]:.3f}", losses[-1] < 0.5 * losses[0])

# 3. agente de evaluacion: interfaz de eval_return.py / eval_bct.py
for topology, kw in VARIANTS:
    for discrete in [False, True]:
        tag = f"{topology}{' discreto' if discrete else ''}"
        kw = dict(dict(topology=topology), **kw)
        cfg = small_cfg(discrete=discrete, **kw)
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

    # enc_n_embd=128: proyeccion aleatoria directo a 128, congelada (su reduce_enc)
    cfg = OmegaConf.load(os.path.join(ROOT, "agent", "bc_hier223_enc128_procgen.yaml")).transformer_cfg
    cfg.pretrained_encoder_path = IMPALA_CKPT
    m = BCARModel(None, 15, cfg)
    enc = m.state_embed
    conv_same = all(torch.equal(v, ckpt["convnet"][k]) for k, v in enc.convnet.state_dict().items())
    frozen = all(not p.requires_grad for p in enc.parameters())
    check(f"IMPALA enc128: conv del checkpoint {conv_same}, proyeccion -> {enc.projection.out_features}, "
          f"congelado {frozen}", conv_same and frozen and enc.projection.out_features == 128)
    pred = m(torch.randint(0, 255, (2, 16, 64, 64, 3), dtype=torch.uint8), torch.randint(0, 15, (2, 16, 1)))
    check(f"IMPALA enc128: forward con pixeles {tuple(pred.shape)}", pred.shape == (2, 16, 15))
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

# 5b. variantes 2+2+3 (seccion 7.15): conteo contra reduce_enc, solo
# informativo; fuera de paridad por diseno (256: ~+40%; enc128: ~-22%, sin la
# fusion por co-atencion ni las proyecciones de su decoder)
for name in ["bc_hier223", "bc_hier223_enc128"]:
    cfg = OmegaConf.load(os.path.join(ROOT, "agent", f"{name}.yaml")).transformer_cfg
    for obs_dim, act_dim in [(17, 6), (24, 6), (78, 12)]:
        m = BCARModel(obs_dim, act_dim, cfg)
        n = sum(p.numel() for p in m.parameters() if p.requires_grad)
        dev = 100 * (n - ref["bc_hier"][obs_dim]) / ref["bc_hier"][obs_dim]
        print(f"[INFO] {name} DMC obs={obs_dim}: {n:,} params ({dev:+.1f}% vs reduce_enc)")
if os.path.exists(IMPALA_CKPT):
    for name in ["bc_uni_procgen", "bc_hier_procgen", "bc_hier223_procgen", "bc_hier223_enc128_procgen"]:
        cfg = OmegaConf.load(os.path.join(ROOT, "agent", f"{name}.yaml")).transformer_cfg
        cfg.pretrained_encoder_path = IMPALA_CKPT
        m = BCARModel(None, 15, cfg)
        n = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"[INFO] {name} CoinRun: {n:,} params entrenables")

# 6. interpretabilidad (eval_sarfa.py / analysis/bc_attn_attr.py): ganchos de
# atencion, logits_for y gradiente integrado en batch
if os.path.exists(IMPALA_CKPT):
    from analysis.bc_attn_attr import attn_attr_layer
    from analysis.trajectory_recorder import load_buffers

    rng = np.random.default_rng(0)
    for name in ["bc_uni_procgen", "bc_hier_procgen", "bc_hier223_enc128_procgen"]:
        cfg = OmegaConf.load(os.path.join(ROOT, "agent", f"{name}.yaml")).transformer_cfg
        cfg.pretrained_encoder_path = IMPALA_CKPT
        ev = BCAREvalAgent((64, 64, 3), (1,), "cpu", sample=False, transformer_cfg=cfg)
        frames = [rng.integers(0, 256, (64, 64, 3), dtype=np.uint8) for _ in range(6)]
        acts = [np.int64(a) for a in rng.integers(0, 15, 5)]
        load_buffers(ev, frames, acts)
        with torch.no_grad():
            ref = ev.mdp(torch.as_tensor(np.stack(frames))[None], torch.as_tensor(np.stack(acts))[None])[0]

        base = ev.logits_for()
        check(f"[{name}] logits_for() = forward sobre los pixeles", torch.allclose(base, ref[-1], atol=1e-4))
        new = rng.integers(0, 256, (3, 64, 64, 3), dtype=np.uint8)
        got = ev.logits_for(obs_override=new, override_index=2)
        ok = True
        for i in range(3):
            fr = list(frames)
            fr[2] = new[i]
            with torch.no_grad():
                r = ev.mdp(torch.as_tensor(np.stack(fr))[None], torch.as_tensor(np.stack(acts))[None])[0, -1]
            ok &= torch.allclose(got[i], r, atol=1e-4)
        check(f"[{name}] logits_for(override) = forward con ese frame reemplazado", ok and got.shape == (3, 15))
        check(f"[{name}] logits_for no modifica los buffers",
              len(ev._obs_buffer) == 6 and np.array_equal(ev._obs_buffer[2], frames[2]))

        a_idx = int(base.argmax())
        n_s = len(ev._obs_buffer)
        for stack, n_layer in ev.mdp.attn_stacks().items():
            L = 2 * n_s if stack == "joint" else n_s
            for layer in range(n_layer):
                with torch.no_grad():
                    pred, att = ev._forward_for_attr(stack, layer, capture_att=True)
                    pred_t, _ = ev._forward_for_attr(stack, layer, tmp_att=att.clone())
                ok = (torch.allclose(pred[0, -1], ref[-1], atol=1e-4) and att.shape == (1, cfg.n_head, L, L)
                      and torch.allclose(pred_t, pred, atol=1e-5))
                check(f"[{name}] {stack}{layer}: captura no cambia la prediccion y tmp_att=att la reproduce", ok)

        # batch de m interpolaciones = bucle de Benjamin (analysis/attn_attr.py)
        stack, layer, m = "joint", ev.mdp.attn_stacks()["joint"] - 1, 8
        batched = attn_attr_layer(ev, stack, layer, a_idx, m=m)
        with torch.no_grad():
            _, att = ev._forward_for_attr(stack, layer, capture_att=True)
        acc = torch.zeros_like(att)
        for k in range(1, m + 1):
            tmp = (k / m * att).clone().requires_grad_(True)
            p, _ = ev._forward_for_attr(stack, layer, tmp_att=tmp)
            p[0, n_s - 1, a_idx].backward()
            acc += tmp.grad
        looped = (att * acc / m)[0]
        check(f"[{name}] IG en batch = IG en bucle", torch.allclose(batched, looped, atol=1e-5, rtol=1e-4))

        # completitud del gradiente integrado: sum(attr) -> F(A) - F(0) al crecer m
        # (la suma de Riemann converge; con m chico puede quedar lejos si F cambia
        # bruscamente cerca de alpha = 0)
        with torch.no_grad():
            f1 = ev._forward_for_attr(stack, layer, tmp_att=att.clone())[0][0, n_s - 1, a_idx]
            f0 = ev._forward_for_attr(stack, layer, tmp_att=torch.zeros_like(att))[0][0, n_s - 1, a_idx]
        diff = (f1 - f0).item()
        err = {mm: abs(attn_attr_layer(ev, stack, layer, a_idx, m=mm).sum().item() - diff) for mm in (50, 1000)}
        check(f"[{name}] IG completitud: |sum attr - (F(A)-F(0))| m=50 {err[50]:.2e} -> m=1000 {err[1000]:.2e} "
              f"(F(A)-F(0) = {diff:.4f})", (err[1000] <= err[50] or err[1000] < 1e-4) and err[1000] < 0.05 * max(abs(diff), 1e-2))
        attr = attn_attr_layer(ev, stack, layer, a_idx, m=20)
        # en la ultima capa conjunta solo la fila del token objetivo s_{n_s-1} llega al logit
        rows = attr.abs().sum(dim=(0, 2))
        check(f"[{name}] ultima capa conjunta: masa solo en la fila objetivo 2(n_s-1)",
              rows[2 * (n_s - 1)] > 0 and rows.sum() - rows[2 * (n_s - 1)] < 1e-8)
else:
    print(f"[SKIP] interpretabilidad: {IMPALA_CKPT} no existe")

print()
if failures:
    print(f"{len(failures)} FALLAS: {failures}")
    sys.exit(1)
print("todo OK")
