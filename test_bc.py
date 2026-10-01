"""
Smoke test standalone para la Etapa BC (METODOLOGIA_DT_HDT.md, seccion 7):
DecisionTransformer / HierarchicalDecisionTransformer con use_rtg=False
(behaviour cloning autoregresivo sin token de return). Tensores dummy, sin
datos reales.
"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from agent.dt import DTAgent, DecisionTransformer
from agent.hdt import HDTAgent, HierarchicalDecisionTransformer

torch.manual_seed(0)

OBS_DIM = 17
ACTION_DIM = 6
NUM_ACTIONS = 15
B = 4
T = 12

failures = []


def check(name, cond):
    status = "OK" if cond else "FAIL"
    print(f"[{status}] {name}")
    if not cond:
        failures.append(name)


def make_config(use_rtg, discrete=False):
    return SimpleNamespace(
        n_embd=32,
        n_head=2,
        n_layer=2,
        n_obs_layer=1,
        n_act_layer=1,
        embd_pdrop=0.0,
        resid_pdrop=0.0,
        attn_pdrop=0.0,
        traj_length=T,
        episode_length=1000,
        return_scale=1000,
        use_rtg=use_rtg,
        discrete_actions=discrete,
        num_actions=NUM_ACTIONS,
    )


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def make_actions(discrete):
    if discrete:
        return torch.randint(0, NUM_ACTIONS, (B, T, 1))
    return torch.rand(B, T, ACTION_DIM) * 2 - 1


for name, Model in [("dt", DecisionTransformer), ("hdt", HierarchicalDecisionTransformer)]:
    for discrete in [False, True]:
        tag = f"{name}{' discreto' if discrete else ''}"
        act_dim = NUM_ACTIONS if discrete else ACTION_DIM
        model = Model(OBS_DIM, act_dim, make_config(use_rtg=False, discrete=discrete)).eval()
        model_rtg = Model(OBS_DIM, act_dim, make_config(use_rtg=True, discrete=discrete))

        obs = torch.randn(B, T, OBS_DIM)
        action = make_actions(discrete)
        ts = torch.arange(T).unsqueeze(0).repeat(B, 1) + 5

        # 1. shape, rtg=None
        pred = model(None, obs, action, timesteps=ts)
        check(f"[{tag}] forward sin rtg: shape", pred.shape == (B, T, act_dim))

        # 2. sin parametros de return; el modelo con rtg tiene exactamente
        #    esos de mas
        names = [n for n, _ in model.named_parameters()]
        check(f"[{tag}] sin return_embed/return_pos_embed", not any("return" in n for n in names))
        extra = n_params(model_rtg) - n_params(model)
        expected = 2 * 32  # return_embed Linear(1, 32)
        if name == "hdt":
            expected += 1000 * 32  # return_pos_embed
        check(f"[{tag}] diferencia de params con rtg = {expected}", extra == expected)

        # 3. el rtg se ignora
        pred2 = model(torch.randn(B, T, 1), obs, action, timesteps=ts)
        check(f"[{tag}] rtg ignorado", torch.allclose(pred, pred2))

        # 4. causalidad: a_k no influye en la prediccion de a_k ni antes;
        #    s_k si influye en la prediccion de a_k
        k = 6
        action_b = action.clone()
        if discrete:
            action_b[:, k:] = (action_b[:, k:] + 1) % NUM_ACTIONS
        else:
            action_b[:, k:] = -action_b[:, k:]
        pred_b = model(None, obs, action_b, timesteps=ts)
        check(f"[{tag}] causal: a_k..a_T no cambian pred[:k+1]",
              torch.allclose(pred[:, : k + 1], pred_b[:, : k + 1], atol=1e-6))
        check(f"[{tag}] causal: a_k si cambia pred[k+1:]",
              not torch.allclose(pred[:, k + 1 :], pred_b[:, k + 1 :], atol=1e-6))
        obs_b = obs.clone()
        obs_b[:, k] += 1.0
        pred_c = model(None, obs_b, action, timesteps=ts)
        check(f"[{tag}] causal: s_k no cambia pred[:k]",
              torch.allclose(pred[:, :k], pred_c[:, :k], atol=1e-6))
        check(f"[{tag}] causal: s_k si cambia pred[k]",
              not torch.allclose(pred[:, k], pred_c[:, k], atol=1e-6))

def utils_to_torch(batch):
    return tuple(torch.as_tensor(x) for x in batch)


def agent_loss(agent, obs, action, ts, mask):
    """loss de update_actor sin dar el paso del optimizador"""
    lr = [g["lr"] for g in agent.opt.param_groups]
    for g in agent.opt.param_groups:
        g["lr"] = 0.0
    loss = agent.update_actor(obs, action, None, None, ts.squeeze(-1).long(), 0, mask=mask)["action_loss"]
    for g, v in zip(agent.opt.param_groups, lr):
        g["lr"] = v
    return loss


# 5. agente: update con batch de 6 (loader sin return_to_go), act sin rtg,
#    y la loss baja sobre un batch fijo
for name, Agent in [("dt", DTAgent), ("hdt", HDTAgent)]:
    for discrete in [False, True]:
        tag = f"{name}{' discreto' if discrete else ''}"
        act_dim = NUM_ACTIONS if discrete else ACTION_DIM
        agent = Agent(
            name=name,
            obs_shape=(OBS_DIM,),
            action_shape=(act_dim,),
            device="cpu",
            lr=1e-3,
            batch_size=B,
            stddev_schedule=0.2,
            use_tb=True,
            transformer_cfg=make_config(use_rtg=False, discrete=discrete),
            use_adamw=False,
        )
        obs = torch.randn(B, T, OBS_DIM)
        action = make_actions(discrete).float()
        reward = torch.randn(B, T, 1)
        discount = torch.ones(B, T, 1)
        ts = (torch.arange(T).unsqueeze(0).repeat(B, 1).unsqueeze(-1)).float()
        batch = tuple(x.numpy() for x in (obs, action, reward, discount, obs, ts))

        def it():
            while True:
                yield batch

        replay_iter = it()
        losses = [agent.update(replay_iter, step)["action_loss"] for step in range(60)]
        check(f"[{tag}] update con batch de 6 elementos, loss baja "
              f"({losses[0]:.3f} -> {losses[-1]:.3f})", losses[-1] < 0.5 * losses[0])

        # batch de 7 = con mascara de relleno (pad_short_episodes): cambiar
        # los targets de los pasos de relleno no cambia la loss
        mask = torch.ones(B, T, 1)
        mask[:, 8:] = 0
        action_pad = make_actions(discrete).float()
        action_pad[:, :8] = action[:, :8]
        agent.train(False)  # sin dropout para comparar
        torch.manual_seed(1)
        o, a_, r, d, no, t = utils_to_torch(batch)
        l1 = agent_loss(agent, o, a_, t, mask)
        l2 = agent_loss(agent, o, action_pad, t, mask)
        l3 = agent_loss(agent, o, action_pad, t, None)
        check(f"[{tag}] mascara: relleno no cambia la loss", abs(l1 - l2) < 1e-5)
        check(f"[{tag}] sin mascara el relleno si cambia la loss", abs(l1 - l3) > 1e-5)
        batch7 = batch + (mask.numpy(),)
        agent.train(True)
        agent.update(iter([batch7]), 0)
        check(f"[{tag}] update con batch de 7 (mascara)", True)

        agent.train(False)
        with torch.no_grad():
            a = agent.act(obs[0].numpy(), action[0].numpy(), None,
                          ts[0, :, 0].long().numpy(), 0)
        expected_shape = (1,) if discrete else (act_dim,)
        check(f"[{tag}] act sin rtg: shape {expected_shape}", tuple(a.shape) == expected_shape)

        if discrete:
            # muestreo con temperatura (eval_coinrun.py --sample): acciones
            # validas y no siempre la misma; temperatura ~0 = argmax
            args = (obs[0].numpy(), action[0].numpy(), None, ts[0, :, 0].long().numpy(), 0)
            with torch.no_grad():
                greedy = int(agent.act(*args)[0])
                agent.sample, agent.temperature = True, 5.0
                draws = {int(agent.act(*args)[0]) for _ in range(200)}
                agent.temperature = 1e-4
                cold = {int(agent.act(*args)[0]) for _ in range(20)}
                agent.sample = False
            check(f"[{tag}] sample T=5: acciones en rango y variadas ({len(draws)} distintas)",
                  all(0 <= d < NUM_ACTIONS for d in draws) and len(draws) > 1)
            check(f"[{tag}] sample T->0 = argmax", cold == {greedy})

print()
if failures:
    print(f"{len(failures)} FALLAS: {failures}")
    sys.exit(1)
print("todo OK")
