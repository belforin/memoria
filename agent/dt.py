import torch
import torch.nn as nn

import utils
from agent.modules.attention import Block


def configure_optimizer(model, lr, weight_decay, warmup_steps):
    """
    AdamW con weight decay separado por grupo de parametros y warmup lineal
    sin decay posterior, igual a la convencion minGPT que usa el codigo
    oficial de Decision Transformer (kzl/decision-transformer,
    gym/experiment.py -> models/decision_transformer.py). Decae en matrices
    de peso (atencion, MLP, proyecciones lineales de entrada); no decae en
    bias, LayerNorm ni embeddings de posicion. Ver METODOLOGIA_DT_HDT.md,
    seccion 2.5.1.
    """
    decay, no_decay = set(), set()
    whitelist_modules = (nn.Linear,)
    blacklist_modules = (nn.LayerNorm, nn.Embedding)
    for mn, m in model.named_modules():
        for pn, _ in m.named_parameters(recurse=False):
            fpn = f"{mn}.{pn}" if mn else pn
            if pn.endswith("bias"):
                no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, whitelist_modules):
                decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, blacklist_modules):
                no_decay.add(fpn)

    param_dict = dict(model.named_parameters())
    assert len(decay & no_decay) == 0, "parametro asignado a ambos grupos"
    assert len(param_dict.keys() - (decay | no_decay)) == 0, (
        "parametro sin asignar a ningun grupo de weight decay"
    )

    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda step: min((step + 1) / warmup_steps, 1.0)
    )
    return opt, scheduler


class DecisionTransformer(nn.Module):
    def __init__(self, obs_dim, action_dim, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.traj_length = config.traj_length
        self.episode_length = config.episode_length
        # cada timestep aporta 3 tokens: return, state, action
        self.max_len = config.traj_length * 3

        # normalizacion z-score de observaciones (media/std del dataset de
        # entrenamiento, fijadas via set_obs_stats antes de entrenar). Sin
        # esto default a identidad (mean=0, std=1) -- igual al codigo
        # oficial de Decision Transformer, que normaliza estados tanto en
        # entrenamiento como en evaluacion. Ver METODOLOGIA_DT_HDT.md,
        # seccion 2.8. Registrado como buffer para que quede en el
        # checkpoint (state_dict) y eval_dt.py recupere los mismos valores
        # usados en entrenamiento.
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_std", torch.ones(obs_dim))

        self.return_embed = nn.Linear(1, self.n_embd)
        self.state_embed = nn.Linear(obs_dim, self.n_embd)
        self.action_embed = nn.Linear(action_dim, self.n_embd)

        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        # DT solo predice la acción, a partir del token de estado
        self.action_head = nn.Sequential(
            nn.LayerNorm(self.n_embd),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_embd, action_dim),
            nn.Tanh(),
        )

        self.initialize_weights()

    def initialize_weights(self):
        # embedding posicional por timestep (aprendido), dimensionado al
        # largo máximo de episodio (no a traj_length, que es solo la
        # ventana de contexto que se muestrea en cada batch)
        self.pos_embed = nn.Embedding(self.episode_length, self.n_embd)
        self.register_buffer(
            "attn_mask",
            torch.tril(torch.ones(self.max_len, self.max_len))[None, None, ...],
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def set_obs_stats(self, mean, std):
        self.obs_mean.copy_(torch.as_tensor(mean, dtype=self.obs_mean.dtype))
        self.obs_std.copy_(torch.as_tensor(std, dtype=self.obs_std.dtype))

    def forward(self, returns_to_go, obs, action, timesteps=None):
        """
        returns_to_go: (B, T, 1)  -- ya escalado (dividido por return_scale)
        obs:           (B, T, obs_dim)
        action:        (B, T, action_dim)
        timesteps:     (B, T) long tensor con el índice real dentro del
                       episodio (si no se pasa, se asume 0..T-1)
        """
        batch_size, T, obs_dim = obs.size()

        if timesteps is None:
            timesteps = (
                torch.arange(T, device=obs.device).unsqueeze(0).repeat(batch_size, 1)
            )
        time_emb = self.pos_embed(timesteps)  # (B, T, n_embd)

        obs = (obs - self.obs_mean) / self.obs_std

        r = self.return_embed(returns_to_go) + time_emb
        s = self.state_embed(obs) + time_emb
        a = self.action_embed(action) + time_emb

        # intercalar como (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        x = (
            torch.stack([r, s, a], dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * T, self.n_embd)
        )

        # CausalSelfAttention ya hace mask[:, :, :T, :T] internamente,
        # así que le pasamos la máscara completa sin recortar acá
        for blk in self.blocks:
            x = blk(x, self.attn_mask)

        # el token de estado (índice 1::3) predice la acción tomada en ese paso
        pred_a = self.action_head(x[:, 1::3])
        return pred_a


class DTAgent:
    def __init__(
        self,
        name,
        obs_shape,
        action_shape,
        device,
        lr,
        batch_size,
        stddev_schedule,
        use_tb,
        transformer_cfg,
        path=None,
        weight_decay=1e-4,
        warmup_steps=10000,
        use_adamw=True,
        obs_mean=None,
        obs_std=None,
    ):
        self.action_dim = action_shape[0]
        self.lr = lr
        self.device = device
        self.use_tb = use_tb
        self.stddev_schedule = stddev_schedule

        payload = None
        if path is not None:
            print("loading existing model...")
            payload = torch.load(path)
            self.config = payload["cfg"]
        else:
            self.config = transformer_cfg

        self.model = DecisionTransformer(
            obs_shape[0], action_shape[0], self.config
        ).to(device)
        if obs_mean is not None:
            self.model.set_obs_stats(obs_mean, obs_std)
        if path is not None:
            # strict=False: checkpoints viejos (previos a set_obs_stats,
            # METODOLOGIA_DT_HDT.md seccion 2.8) no tienen los buffers
            # obs_mean/obs_std -- quedan en su default de identidad
            # (mean=0, std=1), que es exactamente como se entrenaron.
            self.model.load_state_dict(payload["model"], strict=False)

        if use_adamw:
            self.opt, self.scheduler = configure_optimizer(
                self.model, lr, weight_decay, warmup_steps
            )
        else:
            # Adam simple, sin weight decay agrupado ni warmup -- para
            # aislar el efecto de AdamW+warmup en la comparacion de
            # METODOLOGIA_DT_HDT.md seccion 2.8 (scheduler no-op para
            # mantener la misma interfaz en update_actor).
            self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.opt, lr_lambda=lambda step: 1.0
            )
        self.return_scale = self.config.return_scale
        print(
            "number of parameters: %e", sum(p.numel() for p in self.model.parameters())
        )
        self.train()

    def train(self, training=True):
        self.training = training
        self.model.train(training)

    @staticmethod
    def compute_returns_to_go(reward, discount):
        """
        reward, discount: (B, T, 1) -- vienen del batch, paso a paso.
        rtg_t = r_t + discount_t * rtg_{t+1}
        (usa el discount real de cada paso, no un gamma fijo)
        """
        B, T, _ = reward.size()
        rtg = torch.zeros_like(reward)
        running = torch.zeros(B, 1, device=reward.device)
        for t in reversed(range(T)):
            running = reward[:, t] + discount[:, t] * running
            rtg[:, t] = running
        return rtg

    def act(self, obs, action, rtg, timestep, step):
        """
        obs, action, rtg, timestep: buffers de historia con shape (1, T, dim)
        (timestep con shape (1, T), timestep real dentro del episodio).
        rtg acá se pasa SIN escalar (return-to-go crudo); esta función
        se encarga de dividirlo por return_scale antes de pasarlo al modelo,
        igual que se hace en entrenamiento.
        """
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        action = torch.as_tensor(action, device=self.device).unsqueeze(0)
        rtg = torch.as_tensor(rtg, device=self.device).unsqueeze(0) / self.return_scale
        timestep = torch.as_tensor(timestep, device=self.device, dtype=torch.long).unsqueeze(0)
        pred_a = self.model(rtg, obs, action, timesteps=timestep)[:, -1]
        return pred_a.cpu().numpy()[0]

    def update_actor(self, obs, action, reward, discount, timestep, step):
        metrics = dict()

        rtg = self.compute_returns_to_go(reward, discount) / self.return_scale
        pred_a = self.model(rtg, obs, action, timesteps=timestep)

        loss = ((pred_a - action) ** 2).mean()

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        self.scheduler.step()

        if self.use_tb:
            metrics["action_loss"] = loss.item()

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs, timestep = utils.to_torch(
            batch, self.device
        )
        # timestep viene como float desde utils.to_torch; nn.Embedding necesita long
        timestep = timestep.squeeze(-1).long()

        if self.use_tb:
            metrics["batch_reward"] = reward.mean().item()

        metrics.update(
            self.update_actor(obs, action, reward, discount, timestep, step)
        )
        return metrics