"""
Smoke test standalone para el soporte de obs de pixeles + accion discreta
en DecisionTransformer (dt.py) y HierarchicalDecisionTransformer (hdt.py),
agregado en la Etapa 2 (CoinRun) -- ver METODOLOGIA_DT_HDT.md, seccion 3.

Usa un obs_shape sintetico chico (32,32,3) y pretrained_encoder_path=None
(el encoder queda con init aleatoria, entrenable) para no depender del
checkpoint real de 64x64 -- eso se verifica aparte, con datos/checkpoint
reales, antes de encolar los jobs de entrenamiento.
"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn

from agent.dt import DTAgent, DecisionTransformer
from agent.hdt import HDTAgent, HierarchicalDecisionTransformer

torch.manual_seed(0)

OBS_SHAPE = (32, 32, 3)
NUM_ACTIONS = 5
B = 4
TRAJ_LENGTH = 6
EPISODE_LENGTH = 50

dt_config = SimpleNamespace(
    n_embd=16,
    n_head=2,
    n_layer=2,
    embd_pdrop=0.0,
    resid_pdrop=0.0,
    attn_pdrop=0.0,
    traj_length=TRAJ_LENGTH,
    episode_length=EPISODE_LENGTH,
    return_scale=10,
    discrete_actions=True,
    pixel_encoder_type="procgen_impala",
    pretrained_encoder_path=None,
)

hdt_config = SimpleNamespace(
    n_embd=16,
    n_head=2,
    n_obs_layer=2,
    n_act_layer=2,
    n_layer=2,
    traj_length=TRAJ_LENGTH,
    episode_length=EPISODE_LENGTH,
    embd_pdrop=0.0,
    resid_pdrop=0.0,
    attn_pdrop=0.0,
    return_scale=10,
    discrete_actions=True,
    pixel_encoder_type="procgen_impala",
    pretrained_encoder_path=None,
)

failures = []


def check(name, cond):
    status = "OK" if cond else "FAIL"
    print(f"[{status}] {name}")
    if not cond:
        failures.append(name)


def dummy_batch():
    obs = torch.randint(0, 256, (B, TRAJ_LENGTH, *OBS_SHAPE), dtype=torch.uint8)
    action = torch.randint(0, NUM_ACTIONS, (B, TRAJ_LENGTH, 1), dtype=torch.int64)
    rtg = torch.randn(B, TRAJ_LENGTH, 1)
    return obs, action, rtg


# --- DecisionTransformer (pixel obs + accion discreta) ---
dt_model = DecisionTransformer(OBS_SHAPE, NUM_ACTIONS, dt_config)
check("DecisionTransformer.pixel_obs=True", dt_model.pixel_obs)
check("DecisionTransformer.discrete_actions=True", dt_model.discrete_actions)

obs, action, rtg = dummy_batch()
pred_a = dt_model(rtg, obs, action)
check("DT forward shape (B, T, num_actions) logits", pred_a.shape == (B, TRAJ_LENGTH, NUM_ACTIONS))

dt_model.zero_grad()
loss = pred_a.sum()
loss.backward()
no_grad = [n for n, p in dt_model.named_parameters() if p.requires_grad and p.grad is None]
check(f"DT backward llega a todos los parametros entrenables (sin grad: {no_grad})", not no_grad)

# --- HierarchicalDecisionTransformer (pixel obs + accion discreta) ---
hdt_model = HierarchicalDecisionTransformer(OBS_SHAPE, NUM_ACTIONS, hdt_config)
check("HierarchicalDecisionTransformer.pixel_obs=True", hdt_model.pixel_obs)
check("HierarchicalDecisionTransformer.discrete_actions=True", hdt_model.discrete_actions)

obs, action, rtg = dummy_batch()
pred_a = hdt_model(rtg, obs, action)
check("HDT forward shape (B, T, num_actions) logits", pred_a.shape == (B, TRAJ_LENGTH, NUM_ACTIONS))

hdt_model.zero_grad()
loss = pred_a.sum()
loss.backward()
no_grad = [n for n, p in hdt_model.named_parameters() if p.requires_grad and p.grad is None]
check(f"HDT backward llega a todos los parametros entrenables (sin grad: {no_grad})", not no_grad)

# --- DTAgent/HDTAgent: act() y update_actor() de punta a punta ---
for name, agent_cls, config in (("DTAgent", DTAgent, dt_config), ("HDTAgent", HDTAgent, hdt_config)):
    agent = agent_cls(
        name=f"{name}_coinrun_test",
        obs_shape=OBS_SHAPE,
        action_shape=(NUM_ACTIONS,),
        device="cpu",
        lr=1e-4,
        batch_size=B,
        stddev_schedule=0.2,
        use_tb=True,
        transformer_cfg=config,
    )

    obs_hist = torch.randint(0, 256, (TRAJ_LENGTH, *OBS_SHAPE), dtype=torch.uint8).numpy()
    action_hist = torch.randint(0, NUM_ACTIONS, (TRAJ_LENGTH, 1), dtype=torch.int64).numpy()
    rtg_hist = torch.randn(TRAJ_LENGTH, 1).numpy()
    timestep_hist = torch.arange(TRAJ_LENGTH).numpy()
    action_out = agent.act(obs_hist, action_hist, rtg_hist, timestep_hist, step=0)
    check(f"{name}.act devuelve un indice de accion valido", action_out.shape == (1,) and 0 <= int(action_out[0]) < NUM_ACTIONS)

    obs = torch.randint(0, 256, (B, TRAJ_LENGTH, *OBS_SHAPE), dtype=torch.uint8)
    action = torch.randint(0, NUM_ACTIONS, (B, TRAJ_LENGTH, 1), dtype=torch.int64)
    reward = torch.rand(B, TRAJ_LENGTH, 1)
    discount = torch.ones(B, TRAJ_LENGTH, 1)
    timestep = torch.arange(TRAJ_LENGTH).unsqueeze(0).repeat(B, 1)
    metrics = agent.update_actor(obs, action, reward, discount, timestep, step=0)
    action_loss = metrics.get("action_loss")
    check(f"{name}.update_actor produce action_loss (cross-entropy) finito", action_loss is not None and action_loss == action_loss)

    encoder = agent.model.state_embed if name == "DTAgent" else agent.model.obs_encoder.embed
    frozen = all(not p.requires_grad for p in encoder.parameters())
    check(f"{name}: encoder de pixeles SIN pretrained_encoder_path queda entrenable (no congelado)", not frozen)

print()
if failures:
    print(f"{len(failures)} check(s) fallaron: {failures}")
    sys.exit(1)
else:
    print("Todos los checks pasaron.")
