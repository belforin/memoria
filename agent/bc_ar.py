"""
Behaviour cloning autoregresivo (sin return-to-go) enchufado al pipeline de
Benjamin Mancilla: mismos datos (replay_buffer.py), mismos entornos (dmc.py,
Procgen) y mismos scripts de evaluacion (eval_return.py, eval_bct.py), solo
cambia el modelo. Version de la Etapa BC de belforin/memoria
(METODOLOGIA_DT_HDT.md seccion 7): BC-uni = Decision Transformer sin token de
return; BC-hier = Hierarchical Decision Transformer sin token de return.

Topologias (transformer_cfg.topology):
- "uni":  un solo stack causal sobre (s_1, a_1, s_2, a_2, ...).
- "hier": un stack causal por modalidad (n_obs_layer sobre los estados,
          n_act_layer sobre las acciones), despues se intercalan y pasan por
          n_layer bloques causales conjuntos.
En ambas la accion a_t se predice desde el token s_t: ve s_1..s_t y
a_1..a_{t-1}.

Diferencias con agent/dt.py / agent/hdt.py de belforin/memoria, para encajar
en este pipeline sin tocarlo:
- posicion = indice dentro de la ventana (el buffer no entrega el timestep
  del episodio), embedding aprendido de traj_length posiciones;
- sin normalizacion de observaciones (este pipeline no calcula
  estadisticas del dataset; MaskedDP tampoco normaliza);
- bloques de agent/modules/attention.py de este repo (los mismos de
  MaskedDP).

Clases:
- BCARAgent: entrenamiento con pretrain.py (misma interfaz que
  MaskedDPMultimodalAgent: .model, .update(replay_iter, step), .train()).
- BCAREvalAgent: evaluacion con eval_return.py (DMC) y eval_bct.py
  (Procgen): .mdp, .reset(), .act(obs), .eval(); ventana deslizante de K
  pasos, a_t desde s_t en cada paso (closed-loop).
"""
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from agent.modules.attention import Block
from agent.modules.load_pretrained_encoder import load_procgen_impala
from agent.modules.pixel_encoder import PixelEncoder


def _cfg(config, key, default):
    return config.get(key, default) if hasattr(config, "get") else getattr(config, key, default)


class _Stack(nn.Module):
    """n_layer bloques causales sobre una secuencia de largo <= max_len"""

    def __init__(self, config, n_layer, max_len):
        super().__init__()
        self.blocks = nn.ModuleList([Block(config) for _ in range(n_layer)])
        self.register_buffer("attn_mask", torch.tril(torch.ones(max_len, max_len))[None, None])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x, self.attn_mask)
        return x


class BCARModel(nn.Module):
    def __init__(self, obs_dim, action_dim, config):
        super().__init__()
        self.topology = str(_cfg(config, "topology", "uni"))
        assert self.topology in ("uni", "hier"), self.topology
        self.n_embd = config.n_embd
        self.traj_length = config.traj_length
        self.pixel_obs = bool(_cfg(config, "use_pixel_obs", False))
        self.discrete_actions = bool(_cfg(config, "discrete_actions", False))
        self.label_smoothing = float(_cfg(config, "label_smoothing", 0.0))

        if self.pixel_obs:
            self.state_embed = PixelEncoder(
                tuple(config.pixel_obs_shape), self.n_embd,
                encoder_type=str(_cfg(config, "pixel_encoder_type", "procgen_impala")),
            )
        else:
            self.state_embed = nn.Linear(obs_dim, self.n_embd)

        if self.discrete_actions:
            self.action_embed = nn.Embedding(int(config.num_actions), self.n_embd)
            head_out, head_act = int(config.num_actions), []
        else:
            self.action_embed = nn.Linear(action_dim, self.n_embd)
            head_out, head_act = action_dim, [nn.Tanh()]

        T = self.traj_length
        if self.topology == "uni":
            self.pos_embed = nn.Embedding(T, self.n_embd)
            self.stack = _Stack(config, config.n_layer, 2 * T)
        else:
            # cada stream de modalidad tiene su propia posicion, igual que
            # SequenceEncoding en agent/sequence_encoding.py de belforin/memoria
            self.obs_pos_embed = nn.Embedding(T, self.n_embd)
            self.act_pos_embed = nn.Embedding(T, self.n_embd)
            self.obs_stack = _Stack(config, config.n_obs_layer, T)
            self.act_stack = _Stack(config, config.n_act_layer, T)
            self.stack = _Stack(config, config.n_layer, 2 * T)

        self.action_head = nn.Sequential(
            nn.LayerNorm(self.n_embd), nn.ReLU(inplace=True), nn.Linear(self.n_embd, head_out), *head_act
        )

        self.apply(self._init_weights)

        # pesos preentrenados DESPUES de self.apply: si no, el init los pisa
        # (ver METODOLOGIA_DT_HDT.md seccion 3 de belforin/memoria)
        ckpt = _cfg(config, "pretrained_encoder_path", None)
        if self.pixel_obs and ckpt:
            load_procgen_impala(self.state_embed, ckpt, freeze=True)

    @staticmethod
    def _init_weights(m):
        if hasattr(m, "weight") and m.weight is not None and not m.weight.requires_grad:
            return
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def embed_obs(self, obs):
        # obs de pixeles ya pasadas por el encoder congelado (embeddings
        # precalculados): (B, T, n_embd) float, se usan tal cual
        if self.pixel_obs and obs.dim() == 3:
            return obs.float()
        if self.pixel_obs:
            return self.state_embed(obs)
        return self.state_embed(obs.float())

    def embed_actions(self, action):
        if self.discrete_actions:
            a = action.long()
            if a.dim() == 3 and a.size(-1) == 1:
                a = a.squeeze(-1)
            return self.action_embed(a)
        return self.action_embed(action.float())

    def forward(self, obs, action):
        """
        obs:    (B, T, obs_dim) estados, (B, T, H, W, C) uint8 pixeles o
                (B, T, n_embd) embeddings precalculados del encoder
        action: (B, T, action_dim) continua o (B, T[, 1]) indices discretos.
                Puede traer menos pasos que obs (en evaluacion falta a_T);
                se rellena con ceros: la mascara causal impide que a_t
                influya en la prediccion de a_t.
        return: (B, T, action_dim) acciones o (B, T, num_actions) logits
        """
        B, T = obs.shape[0], obs.shape[1]
        s = self.embed_obs(obs)
        a = self.embed_actions(action) if action.shape[1] > 0 else s.new_zeros(B, 0, self.n_embd)
        if a.shape[1] < T:
            a = torch.cat([a, a.new_zeros(B, T - a.shape[1], self.n_embd)], dim=1)

        pos = torch.arange(T, device=s.device)
        if self.topology == "uni":
            s = s + self.pos_embed(pos)
            a = a + self.pos_embed(pos)
        else:
            s = self.obs_stack(s + self.obs_pos_embed(pos))
            a = self.act_stack(a + self.act_pos_embed(pos))

        x = torch.stack([s, a], dim=2).reshape(B, 2 * T, self.n_embd)
        x = self.stack(x)
        return self.action_head(x[:, 0::2])

    def action_loss(self, pred, action, mask=None):
        """mask: (B, T, 1) 1 en pasos reales, 0 en relleno (episodios cortos)"""
        B, T = pred.shape[:2]
        if self.discrete_actions:
            tgt = action.long()
            if tgt.dim() == 3 and tgt.size(-1) == 1:
                tgt = tgt.squeeze(-1)
            per_step = F.cross_entropy(
                pred.reshape(B * T, -1), tgt.reshape(B * T),
                label_smoothing=self.label_smoothing, reduction="none",
            ).reshape(B, T)
        else:
            per_step = ((pred - action.float()) ** 2).mean(dim=-1)
        if mask is None:
            return per_step.mean()
        m = mask.reshape(B, T).float()
        return (per_step * m).sum() / m.sum().clamp(min=1.0)


def _dims(obs_shape, action_shape, config):
    obs_dim = None if _cfg(config, "use_pixel_obs", False) else int(obs_shape[0])
    if _cfg(config, "discrete_actions", False):
        action_dim = int(config.num_actions)
    else:
        action_dim = int(action_shape[0])
    return obs_dim, action_dim


class BCARAgent:
    def __init__(self, name, obs_shape, action_shape, device, lr, batch_size, use_tb,
                 transformer_cfg, **kwargs):
        self.name = name
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.use_tb = use_tb
        self.config = transformer_cfg
        obs_dim, action_dim = _dims(obs_shape, action_shape, transformer_cfg)
        self.action_dim = action_dim

        self.model = BCARModel(obs_dim, action_dim, transformer_cfg).to(device)

        # AdamW con weight decay solo en matrices 2D (Linear/Conv), igual que
        # MaskedDPMultimodalAgent._classify_params; con weight_decay=0 (la
        # config de Benjamin) equivale a Adam.
        wd = float(_cfg(transformer_cfg, "weight_decay", 0.0))
        decay, no_decay = [], []
        for module in self.model.modules():
            for pn, p in module.named_parameters(recurse=False):
                if not p.requires_grad:
                    continue
                if pn.endswith("weight") and isinstance(module, (nn.Linear, nn.Conv2d)):
                    decay.append(p)
                else:
                    no_decay.append(p)
        self.opt = torch.optim.AdamW(
            [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}], lr=lr
        )

        n_total = sum(p.numel() for p in self.model.parameters())
        n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[BCARAgent] topology={self.model.topology} | params: {n_total:,} total, "
              f"{n_train:,} entrenables")
        self.train()

    def train(self, training=True):
        self.training = training
        self.model.train(training)

    def update(self, replay_iter, step=None):
        batch = next(replay_iter)
        obs, action, _, _, _, mask = utils.to_torch(batch, self.device)
        pred = self.model(obs, action)
        loss = self.model.action_loss(pred, action, mask)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()

        metrics = dict()
        if self.use_tb:
            metrics["action_loss"] = loss.item()
        return metrics


class BCAREvalAgent:
    """
    Closed-loop: en cada paso agrega s_t a la ventana (ultimos K estados y
    K-1 acciones), predice a_t desde s_t y la ejecuta.

    Mismos argumentos que MaskingEvalAgentMultimodal (eval_return.py lee
    T_cond/T_pred/replan_freq de la config) y BCTEvalAgentMultimodal
    (eval_bct.py lee K/temperature/sample). K = largo de la ventana
    (default traj_length); T_cond/T_pred/replan_freq no se usan: el modelo
    autoregresivo siempre actua paso a paso.
    """

    def __init__(self, obs_shape, action_shape, device, K=None, temperature=1.0, sample=False,
                 path=None, transformer_cfg=None, **kwargs):
        self.device = device
        self.temperature = temperature
        self.sample = sample

        payload = None
        if path is not None:
            print(f"[BCAREvalAgent] Loading snapshot: {path}")
            payload = torch.load(path, map_location=device)
            self.config = payload["cfg"]
        else:
            assert transformer_cfg is not None, "Must provide either `path` or `transformer_cfg`."
            self.config = transformer_cfg

        obs_dim, action_dim = _dims(obs_shape, action_shape, self.config)
        self.mdp = BCARModel(obs_dim, action_dim, self.config).to(device)
        if payload is not None:
            self.mdp.load_state_dict(payload["model"])
        for p in self.mdp.parameters():
            p.requires_grad = False
        self.mdp.eval()

        self.K = int(K) if K is not None else int(self.config.traj_length)
        assert 1 <= self.K <= int(self.config.traj_length), (self.K, self.config.traj_length)
        print(f"[BCAREvalAgent] topology={self.mdp.topology} | K={self.K} | "
              f"discrete={self.mdp.discrete_actions} | sample={self.sample} | "
              f"temperature={self.temperature}")
        self._obs_buffer = None
        self._action_buffer = None

    def reset(self):
        self._obs_buffer = deque(maxlen=self.K)
        self._action_buffer = deque(maxlen=self.K - 1)

    def eval(self):
        self.mdp.eval()

    def act(self, obs):
        if self.mdp.pixel_obs and obs.ndim == 4:  # (1, H, W, C) de Procgen
            obs = obs[0]
        self._obs_buffer.append(obs)

        obs_t = torch.as_tensor(np.stack(self._obs_buffer), device=self.device).unsqueeze(0)
        if len(self._action_buffer) > 0:
            act_t = torch.as_tensor(np.stack(self._action_buffer), device=self.device).unsqueeze(0)
        elif self.mdp.discrete_actions:
            act_t = torch.zeros(1, 0, dtype=torch.long, device=self.device)
        else:
            act_t = torch.zeros(1, 0, self.mdp.action_head[2].out_features, device=self.device)

        with torch.no_grad():
            pred = self.mdp(obs_t, act_t)[0, -1]

        if self.mdp.discrete_actions:
            if self.sample:
                dist = torch.distributions.Categorical(logits=pred / self.temperature)
                a = int(dist.sample().item())
            else:
                a = int(torch.argmax(pred).item())
            self._action_buffer.append(np.int64(a))
            return np.array([a], dtype=np.int64)

        action = pred.cpu().numpy().astype(np.float32)
        self._action_buffer.append(action)
        return action
