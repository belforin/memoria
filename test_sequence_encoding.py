"""
Smoke test standalone para agent/sequence_encoding.py.
Corre con tensores dummy (sin datos reales), pensado para validar antes de
integrar los encoders en mdp.py/mdp_rl.py.
"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from agent.sequence_encoding import (
    ActionSequenceEncoding,
    ObservationSequenceEncoding,
    SequenceEncoding,
)

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
    traj_length=TRAJ_LENGTH,
    episode_length=EPISODE_LENGTH,
    attn_pdrop=0.0,
    resid_pdrop=0.0,
)

failures = []


def check(name, cond):
    status = "OK" if cond else "FAIL"
    print(f"[{status}] {name}")
    if not cond:
        failures.append(name)


# 1. shapes básicos, con y sin timesteps explícito
obs_enc = ObservationSequenceEncoding(OBS_DIM, config)
action_enc = ActionSequenceEncoding(ACTION_DIM, config)

obs = torch.randn(B, TRAJ_LENGTH, OBS_DIM)
action = torch.randn(B, TRAJ_LENGTH, ACTION_DIM)

out_obs_default = obs_enc(obs)
check(
    "ObservationSequenceEncoding shape (timesteps=None)",
    out_obs_default.shape == (B, TRAJ_LENGTH, config.n_embd),
)

real_timesteps = torch.randint(0, EPISODE_LENGTH - TRAJ_LENGTH, (B, 1)) + torch.arange(
    TRAJ_LENGTH
)
out_obs_real_t = obs_enc(obs, timesteps=real_timesteps)
check(
    "ObservationSequenceEncoding shape (timesteps reales)",
    out_obs_real_t.shape == (B, TRAJ_LENGTH, config.n_embd),
)

out_action = action_enc(action, timesteps=real_timesteps)
check(
    "ActionSequenceEncoding shape (timesteps reales)",
    out_action.shape == (B, TRAJ_LENGTH, config.n_embd),
)

# 2. pos_embed realmente cambia la salida (timesteps=None vs timesteps reales
#    con offset != 0..T-1)
check(
    "timesteps reales producen salida distinta a timesteps=None (offset != 0)",
    not torch.allclose(out_obs_default, out_obs_real_t),
)

# 3. dos instancias no comparten el mismo objeto Parameter (identidad, no valor:
#    los bias arrancan en 0 en ambos por diseño, así que comparar por valor daría
#    un falso positivo)
shared_params = any(
    p1 is p2
    for p1 in obs_enc.parameters()
    for p2 in action_enc.parameters()
)
check("obs_enc y action_enc no comparten pesos", not shared_params)

# 4. límite de episode_length: índice válido (episode_length - 1) no debe fallar
try:
    t_max = torch.full((B, TRAJ_LENGTH), EPISODE_LENGTH - 1, dtype=torch.long)
    obs_enc(obs, timesteps=t_max)
    check("timestep == episode_length - 1 no revienta", True)
except IndexError:
    check("timestep == episode_length - 1 no revienta", False)

# 5. índice fuera de rango sí debe fallar (documenta el invariante, no es bug)
try:
    t_oob = torch.full((B, TRAJ_LENGTH), EPISODE_LENGTH, dtype=torch.long)
    obs_enc(obs, timesteps=t_oob)
    check("timestep == episode_length lanza IndexError (esperado)", False)
except IndexError:
    check("timestep == episode_length lanza IndexError (esperado)", True)

# 6. contexto parcial (T < traj_length)
obs_short = torch.randn(B, 5, OBS_DIM)
out_short = obs_enc(obs_short)
check("forward con T < traj_length funciona", out_short.shape == (B, 5, config.n_embd))

# 7. causalidad: cambiar el último timestep no debe afectar la salida de
#    timesteps anteriores (self-attention causal real, no solo declarada)
obs_enc.eval()
with torch.no_grad():
    x1 = torch.randn(1, TRAJ_LENGTH, OBS_DIM)
    x2 = x1.clone()
    x2[:, -1] += 100.0  # perturbación grande solo en el último timestep

    out1 = obs_enc(x1)
    out2 = obs_enc(x2)

check(
    "causalidad: perturbar el último timestep no cambia los anteriores",
    torch.allclose(out1[:, :-1], out2[:, :-1], atol=1e-5),
)
check(
    "causalidad: perturbar el último timestep SÍ cambia su propia salida",
    not torch.allclose(out1[:, -1], out2[:, -1]),
)
obs_enc.train()

# 8. backward: el gradiente llega a todos los parámetros (embed, blocks, pos_embed)
obs_enc.zero_grad()
out = obs_enc(obs, timesteps=real_timesteps)
loss = out.sum()
loss.backward()

no_grad_params = [
    name for name, p in obs_enc.named_parameters() if p.grad is None
]
check(f"backward llega a todos los parámetros (sin grad: {no_grad_params})", not no_grad_params)

print()
if failures:
    print(f"{len(failures)} check(s) fallaron: {failures}")
    sys.exit(1)
else:
    print("Todos los checks pasaron.")
