"""
Smoke test standalone para agent/dt.py (Decision Transformer).
Corre con tensores dummy (sin datos reales) y, si ya se corrio
d4rl_data.py, tambien con un batch real de D4RL (halfcheetah-expert).
"""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

import utils
from agent.dt import DTAgent, DecisionTransformer

torch.manual_seed(0)

OBS_DIM = 17
ACTION_DIM = 6
B = 4
TRAJ_LENGTH = 12
EPISODE_LENGTH = 1000

config = SimpleNamespace(
    n_embd=32,
    n_head=2,
    n_layer=2,
    embd_pdrop=0.0,
    resid_pdrop=0.0,
    attn_pdrop=0.0,
    traj_length=TRAJ_LENGTH,
    episode_length=EPISODE_LENGTH,
    return_scale=1000,
)

failures = []


def check(name, cond):
    status = "OK" if cond else "FAIL"
    print(f"[{status}] {name}")
    if not cond:
        failures.append(name)


model = DecisionTransformer(OBS_DIM, ACTION_DIM, config)

obs = torch.randn(B, TRAJ_LENGTH, OBS_DIM)
action = torch.randn(B, TRAJ_LENGTH, ACTION_DIM)
rtg = torch.randn(B, TRAJ_LENGTH, 1)

# 1. shape básico, timesteps=None
pred_default = model(rtg, obs, action)
check(
    "forward shape (timesteps=None)",
    pred_default.shape == (B, TRAJ_LENGTH, ACTION_DIM),
)

# 2. shape con timesteps reales (offset != 0..T-1, como los que entrega el
#    replay buffer)
real_timesteps = torch.randint(0, EPISODE_LENGTH - TRAJ_LENGTH, (B, 1)) + torch.arange(
    TRAJ_LENGTH
)
pred_real_t = model(rtg, obs, action, timesteps=real_timesteps)
check(
    "forward shape (timesteps reales)",
    pred_real_t.shape == (B, TRAJ_LENGTH, ACTION_DIM),
)

# 3. el pos_embed realmente afecta la salida
check(
    "timesteps reales producen salida distinta a timesteps=None (offset != 0)",
    not torch.allclose(pred_default, pred_real_t),
)

# 4. límite de episode_length: índice válido no debe fallar, uno fuera de
#    rango sí
try:
    t_max = torch.full((B, TRAJ_LENGTH), EPISODE_LENGTH - 1, dtype=torch.long)
    model(rtg, obs, action, timesteps=t_max)
    check("timestep == episode_length - 1 no revienta", True)
except IndexError:
    check("timestep == episode_length - 1 no revienta", False)

try:
    t_oob = torch.full((B, TRAJ_LENGTH), EPISODE_LENGTH, dtype=torch.long)
    model(rtg, obs, action, timesteps=t_oob)
    check("timestep == episode_length lanza IndexError (esperado)", False)
except IndexError:
    check("timestep == episode_length lanza IndexError (esperado)", True)

# 5. contexto parcial (T < traj_length)
obs_short = torch.randn(B, 5, OBS_DIM)
action_short = torch.randn(B, 5, ACTION_DIM)
rtg_short = torch.randn(B, 5, 1)
pred_short = model(rtg_short, obs_short, action_short)
check(
    "forward con T < traj_length funciona",
    pred_short.shape == (B, 5, ACTION_DIM),
)

# 6. causalidad de punta a punta: perturbar el último timestep de obs/action
#    no debe cambiar la predicción en timesteps anteriores
model.eval()
with torch.no_grad():
    obs1 = torch.randn(1, TRAJ_LENGTH, OBS_DIM)
    action1 = torch.randn(1, TRAJ_LENGTH, ACTION_DIM)
    rtg1 = torch.randn(1, TRAJ_LENGTH, 1)

    obs2 = obs1.clone()
    action2 = action1.clone()
    rtg2 = rtg1.clone()
    obs2[:, -1] += 100.0
    action2[:, -1] += 100.0

    pred1 = model(rtg1, obs1, action1)
    pred2 = model(rtg2, obs2, action2)

check(
    "causalidad: perturbar el último timestep no cambia predicciones anteriores",
    torch.allclose(pred1[:, :-1], pred2[:, :-1], atol=1e-5),
)
check(
    "causalidad: perturbar el último timestep SÍ cambia su propia predicción",
    not torch.allclose(pred1[:, -1], pred2[:, -1]),
)
model.train()

# 7. backward: el gradiente llega a todos los parámetros
model.zero_grad()
out = model(rtg, obs, action, timesteps=real_timesteps)
loss = out.sum()
loss.backward()

no_grad_params = [name for name, p in model.named_parameters() if p.grad is None]
check(f"backward llega a todos los parámetros (sin grad: {no_grad_params})", not no_grad_params)

# 8. smoke test de DTAgent: instanciar el agente completo y ejercitar
#    act()/update_actor() con tensores dummy
agent = DTAgent(
    name="dt_test",
    obs_shape=(OBS_DIM,),
    action_shape=(ACTION_DIM,),
    device="cpu",
    lr=1e-4,
    batch_size=B,
    stddev_schedule=0.2,
    use_tb=True,
    transformer_cfg=config,
)

hist_obs = torch.randn(TRAJ_LENGTH, OBS_DIM).numpy()
hist_action = torch.randn(TRAJ_LENGTH, ACTION_DIM).numpy()
hist_rtg = torch.randn(TRAJ_LENGTH, 1).numpy()
hist_timestep = real_timesteps[0].numpy()

try:
    with torch.no_grad():
        act_out = agent.act(hist_obs, hist_action, hist_rtg, hist_timestep, step=0)
    check("DTAgent.act devuelve un array de forma (action_dim,)", act_out.shape == (ACTION_DIM,))
except Exception as e:
    check(f"DTAgent.act no revienta ({e!r})", False)

batch_obs = torch.randn(B, TRAJ_LENGTH, OBS_DIM)
batch_action = torch.randn(B, TRAJ_LENGTH, ACTION_DIM)
batch_reward = torch.randn(B, TRAJ_LENGTH, 1)
batch_discount = torch.ones(B, TRAJ_LENGTH, 1)
batch_timestep = real_timesteps

try:
    metrics = agent.update_actor(
        batch_obs, batch_action, batch_reward, batch_discount, batch_timestep, step=0
    )
    loss_val = metrics.get("action_loss")
    check(
        "DTAgent.update_actor produce action_loss finito",
        loss_val is not None and loss_val == loss_val,  # NaN != NaN
    )
except Exception as e:
    check(f"DTAgent.update_actor no revienta ({e!r})", False)

# 9. datos reales de D4RL (halfcheetah-expert), si ya se corrió d4rl_data.py
DATA_DIR = Path(__file__).parent / "data" / "halfcheetah_expert" / "halfcheetah"
if DATA_DIR.exists():
    from replay_buffer import make_replay_loader

    loader = make_replay_loader(
        None,
        DATA_DIR,
        max_size=100_000,
        batch_size=B,
        num_workers=0,
        discount=0.99,
        domain="halfcheetah",
        traj_length=TRAJ_LENGTH,
        relabel=False,
    )
    try:
        real_batch = next(iter(loader))
        real_obs, real_action, real_reward, real_discount, real_next_obs, real_timestep = (
            utils.to_torch(real_batch, "cpu")
        )
        real_timestep = real_timestep.squeeze(-1).long()
        check(
            "batch real de D4RL tiene las dimensiones esperadas (obs 17, action 6)",
            real_obs.shape == (B, TRAJ_LENGTH, OBS_DIM)
            and real_action.shape == (B, TRAJ_LENGTH, ACTION_DIM),
        )
        metrics = agent.update_actor(
            real_obs, real_action, real_reward, real_discount, real_timestep, step=0
        )
        real_loss = metrics.get("action_loss")
        check(
            "DTAgent.update_actor entrena con un batch real de D4RL (loss finito)",
            real_loss is not None and real_loss == real_loss,
        )
    except Exception as e:
        check(f"DTAgent entrena con datos reales de D4RL sin reventar ({e!r})", False)
else:
    print(
        f"[SKIP] no se encontraron datos reales en {DATA_DIR} "
        "(correr d4rl_data.py primero)"
    )

print()
if failures:
    print(f"{len(failures)} check(s) fallaron: {failures}")
    sys.exit(1)
else:
    print("Todos los checks pasaron.")
