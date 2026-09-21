import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from agent.modules.attention import Block
from agent.modules.pixel_encoder import PixelEncoder
from agent.modules.load_pretrained_encoder import load_procgen_impala


def configure_optimizer(model, lr, weight_decay, warmup_steps):
    """
    AdamW con weight decay separado por grupo de parametros y warmup lineal
    sin decay posterior, igual a la convencion minGPT que usa el codigo
    oficial de Decision Transformer (kzl/decision-transformer,
    gym/experiment.py -> models/decision_transformer.py). Decae en matrices
    de peso (atencion, MLP, proyecciones lineales de entrada); no decae en
    bias, LayerNorm ni embeddings de posicion. Ver METODOLOGIA_DT_HDT.md,
    seccion 2.5.1.

    Parametros con requires_grad=False (el encoder visual congelado de la
    Etapa 2, ver seccion 3) se excluyen por completo de la clasificacion:
    nunca reciben gradiente, así que no necesitan grupo de weight decay, y
    ademas rompen la clasificacion por tipo de capa (el encoder IMPALA
    tiene nn.Conv2d, que no es ni nn.Linear ni nn.LayerNorm/nn.Embedding).
    Mismo fix que agent/mdp.py::_classify_params de Benjamin Mancilla
    (rama upstream/hier-procgen) resuelve para su encoder congelado.
    """
    decay, no_decay = set(), set()
    # nn.Conv2d (y Conv1d/Conv3d, por si acaso): el encoder visual de la
    # Etapa 2 (agent/modules/impala_cnn.py) es todo Conv2d -- si alguna vez
    # se entrena sin congelar (encoder_trainable, no usado hoy), sus pesos
    # deben caer en el grupo "decay" igual que cualquier matriz de pesos.
    # Mismos decay_modules que agent/mdp.py::_classify_params de Benjamin
    # Mancilla (rama upstream/hier-procgen).
    whitelist_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)
    blacklist_modules = (nn.LayerNorm, nn.Embedding)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            fpn = f"{mn}.{pn}" if mn else pn
            if pn.endswith("bias"):
                no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, whitelist_modules):
                decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, blacklist_modules):
                no_decay.add(fpn)

    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
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
    def __init__(self, obs_shape, action_dim, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.traj_length = config.traj_length
        self.episode_length = config.episode_length
        # cada timestep aporta 3 tokens: return, state, action
        self.max_len = config.traj_length * 3

        # obs_shape: int (o tupla de 1) = obs vectorial (D4RL, Etapa 1);
        # tupla de 3 (H,W,C) = obs de pixeles (CoinRun, Etapa 2). Ver
        # METODOLOGIA_DT_HDT.md, seccion 3.
        if isinstance(obs_shape, int):
            obs_shape = (obs_shape,)
        self.pixel_obs = len(obs_shape) == 3

        # acción discreta (Procgen, 15 acciones) vs continua (D4RL). Ver
        # METODOLOGIA_DT_HDT.md, seccion 3 -- default False no rompe
        # configs/tests existentes de D4RL.
        self.discrete_actions = bool(getattr(config, "discrete_actions", False))
        self.label_smoothing = float(getattr(config, "label_smoothing", 0.0))

        # normalizacion z-score de observaciones (media/std del dataset de
        # entrenamiento, fijadas via set_obs_stats antes de entrenar). Sin
        # esto default a identidad (mean=0, std=1) -- igual al codigo
        # oficial de Decision Transformer, que normaliza estados tanto en
        # entrenamiento como en evaluacion. Ver METODOLOGIA_DT_HDT.md,
        # seccion 2.8. Registrado como buffer para que quede en el
        # checkpoint (state_dict) y eval_dt.py recupere los mismos valores
        # usados en entrenamiento. Se registra siempre (incluso con obs de
        # pixeles, donde forward() lo ignora) para no bifurcar el
        # state_dict/checkpoint segun el tipo de obs.
        obs_dim = obs_shape[0] if not self.pixel_obs else 1
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_std", torch.ones(obs_dim))

        self.return_embed = nn.Linear(1, self.n_embd)
        if self.pixel_obs:
            # PixelEncoder/load_procgen_impala: mismo patron que
            # agent/mdp.py de Benjamin Mancilla (rama upstream/hier-procgen).
            # El checkpoint pretrained se carga MAS ABAJO, despues de
            # self.initialize_weights() -- ver nota ahi (orden importa).
            pixel_encoder_type = str(getattr(config, "pixel_encoder_type", "procgen_impala"))
            self.state_embed = PixelEncoder(obs_shape, self.n_embd, encoder_type=pixel_encoder_type)
        else:
            self.state_embed = nn.Linear(obs_shape[0], self.n_embd)

        if self.discrete_actions:
            num_actions = action_dim
            self.action_embed = nn.Embedding(num_actions, self.n_embd)
        else:
            self.action_embed = nn.Linear(action_dim, self.n_embd)

        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        # DT solo predice la acción, a partir del token de estado
        if self.discrete_actions:
            # logits crudos -- softmax aplicado implicito dentro de
            # F.cross_entropy (DTAgent.update_actor), no aca. Mismo patron
            # que agent/mdp.py::action_head discreto.
            self.action_head = nn.Sequential(
                nn.LayerNorm(self.n_embd),
                nn.ReLU(inplace=True),
                nn.Linear(self.n_embd, action_dim),
            )
        else:
            self.action_head = nn.Sequential(
                nn.LayerNorm(self.n_embd),
                nn.ReLU(inplace=True),
                nn.Linear(self.n_embd, action_dim),
                nn.Tanh(),
            )

        self.initialize_weights()

        if self.pixel_obs:
            # Cargar (y congelar) los pesos pretrained DESPUES de
            # initialize_weights(): self.apply(self._init_weights) de mas
            # arriba reinicializa con xavier_uniform_ CUALQUIER nn.Linear
            # del modelo, incluida state_embed.projection -- si se carga
            # el checkpoint antes, ese paso lo pisa. Bug real encontrado en
            # agent/mdp.py de Benjamin (rama upstream/hier-procgen): en su
            # caso queda enmascarado porque su enc_n_embd=128 != 256 (dim
            # del checkpoint) fuerza el fallback ignore_proj=True, asi que
            # la proyeccion nunca se cargaba en primer lugar. Aca n_embd=256
            # coincide exacto con el checkpoint, asi que el orden si importa.
            # Ver METODOLOGIA_DT_HDT.md, seccion 3.
            ckpt_path = getattr(config, "pretrained_encoder_path", None)
            if ckpt_path is not None:
                load_procgen_impala(self.state_embed, ckpt_path, freeze=True)

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
        # Si m.weight ya esta congelado (requires_grad=False), es un
        # submodulo con pesos pretrained (encoder visual de la Etapa 2) --
        # no reinicializar. self.apply() recorre TODO el arbol de
        # submodulos, asi que sin este guard pisaria pesos pretrained ya
        # cargados. Ver METODOLOGIA_DT_HDT.md, seccion 3.
        if hasattr(m, "weight") and m.weight is not None and not m.weight.requires_grad:
            return
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
        obs:           (B, T, obs_dim) obs vectorial, o (B, T, H, W, C) uint8
                       obs de pixeles (self.pixel_obs)
        action:        (B, T, action_dim) continua, o (B, T, 1) int64 indices
                       de accion (self.discrete_actions)
        timesteps:     (B, T) long tensor con el índice real dentro del
                       episodio (si no se pasa, se asume 0..T-1)
        """
        batch_size, T = obs.shape[0], obs.shape[1]

        if timesteps is None:
            timesteps = (
                torch.arange(T, device=obs.device).unsqueeze(0).repeat(batch_size, 1)
            )
        time_emb = self.pos_embed(timesteps)  # (B, T, n_embd)

        if not self.pixel_obs:
            obs = (obs - self.obs_mean) / self.obs_std

        r = self.return_embed(returns_to_go) + time_emb
        s = self.state_embed(obs) + time_emb

        if self.discrete_actions:
            # (B, T, 1) o (B, T) int -> (B, T) long, igual que
            # agent/mdp.py::forward_encoder de Benjamin (rama
            # upstream/hier-procgen).
            a_idx = action.long()
            if a_idx.dim() == 3 and a_idx.size(-1) == 1:
                a_idx = a_idx.squeeze(-1)
            a = self.action_embed(a_idx) + time_emb
        else:
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
            obs_shape, action_shape[0], self.config
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
        if self.model.discrete_actions:
            # logits (1, num_actions) -> indice de accion escalar. argmax
            # (greedy), no muestreo -- consistente con --sample=False de
            # eval_coinrun.py por defecto; ver METODOLOGIA_DT_HDT.md secc. 3.
            action_idx = pred_a.argmax(dim=-1)
            return action_idx.cpu().numpy()
        return pred_a.cpu().numpy()[0]

    def update_actor(self, obs, action, reward, discount, timestep, step):
        metrics = dict()

        rtg = self.compute_returns_to_go(reward, discount) / self.return_scale
        pred_a = self.model(rtg, obs, action, timesteps=timestep)

        if self.model.discrete_actions:
            # logits (B, T, num_actions) vs. indices de accion (B, T, 1) o
            # (B, T) -- mismo squeeze/reshape que agent/mdp.py::forward_loss
            # de Benjamin (rama upstream/hier-procgen).
            B, T, A = pred_a.shape
            a_tgt = action.long()
            if a_tgt.dim() == 3 and a_tgt.size(-1) == 1:
                a_tgt = a_tgt.squeeze(-1)
            loss = F.cross_entropy(
                pred_a.reshape(B * T, A),
                a_tgt.reshape(B * T),
                label_smoothing=self.model.label_smoothing,
            )
        else:
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