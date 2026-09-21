from types import SimpleNamespace

import torch
import torch.nn as nn

from agent.dt import DTAgent, configure_optimizer
from agent.modules.attention import Block
from agent.sequence_encoding import ActionSequenceEncoding, ObservationSequenceEncoding

_SHARED_FIELDS = (
    "n_embd",
    "n_head",
    "traj_length",
    "episode_length",
    "embd_pdrop",
    "resid_pdrop",
    "attn_pdrop",
)

# Campos de la Etapa 2 (visual, CoinRun -- METODOLOGIA_DT_HDT.md seccion 3),
# opcionales: los config de D4RL (dt.yaml/hdt.yaml) no los tienen, así que
# se leen con getattr(..., default) en vez de exigirlos como _SHARED_FIELDS.
_OPTIONAL_FIELDS = {
    "discrete_actions": False,
    "pixel_encoder_type": "procgen_impala",
    "pretrained_encoder_path": None,
}


def _sub_config(config, n_layer):
    """
    SequenceEncoding/Block leen `config.n_layer` directamente, pero HDT
    necesita tres profundidades distintas (n_obs_layer, n_act_layer,
    n_layer) a partir de un único transformer_cfg compartido. Arma un
    namespace liviano con los campos compartidos copiados y n_layer
    sobreescrito. getattr() funciona igual sobre un DictConfig de
    OmegaConf (entrenamiento real) que sobre un SimpleNamespace (tests).
    """
    fields = {f: getattr(config, f) for f in _SHARED_FIELDS}
    fields.update({f: getattr(config, f, default) for f, default in _OPTIONAL_FIELDS.items()})
    ns = SimpleNamespace(**fields)
    ns.n_layer = n_layer
    return ns


class HierarchicalDecisionTransformer(nn.Module):
    """
    Igual que DecisionTransformer (dt.py), pero los tokens de estado y
    acción no son una proyección lineal de un solo timestep: vienen de
    T_obs/T_act, dos transformers autoregresivos separados (uno por
    modalidad) que ya contextualizan cada secuencia con su propia
    atención causal antes de intercalarla con el token de return.
    """

    def __init__(self, obs_shape, action_dim, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.traj_length = config.traj_length
        self.episode_length = config.episode_length
        self.max_len = config.traj_length * 3

        # obs_shape: int (o tupla de 1) = obs vectorial (D4RL, Etapa 1);
        # tupla de 3 (H,W,C) = obs de pixeles (CoinRun, Etapa 2). Ver
        # METODOLOGIA_DT_HDT.md, seccion 3.
        if isinstance(obs_shape, int):
            obs_shape = (obs_shape,)
        self.pixel_obs = len(obs_shape) == 3
        self.discrete_actions = bool(getattr(config, "discrete_actions", False))
        self.label_smoothing = float(getattr(config, "label_smoothing", 0.0))

        obs_config = _sub_config(config, config.n_obs_layer)
        action_config = _sub_config(config, config.n_act_layer)
        top_config = _sub_config(config, config.n_layer)

        # normalizacion z-score de observaciones, igual que dt.py -- ver
        # METODOLOGIA_DT_HDT.md, seccion 2.8. Se registra siempre (incluso
        # con obs de pixeles, donde forward() lo ignora), ver nota
        # equivalente en agent/dt.py.
        obs_dim = obs_shape[0] if not self.pixel_obs else 1
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_std", torch.ones(obs_dim))

        self.obs_encoder = ObservationSequenceEncoding(obs_shape, obs_config)
        self.action_encoder = ActionSequenceEncoding(action_dim, action_config)

        self.return_embed = nn.Linear(1, self.n_embd)

        self.blocks = nn.ModuleList([Block(top_config) for _ in range(top_config.n_layer)])

        if self.discrete_actions:
            # logits crudos, mismo patron que agent/dt.py -- ver ahí.
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

    def initialize_weights(self):
        # T_obs/T_act ya traen su propio pos_embed interno (aplicado sobre
        # su salida contextualizada). El token de return no pasa por
        # ningún encoder, así que necesita uno propio acá.
        self.return_pos_embed = nn.Embedding(self.episode_length, self.n_embd)
        self.register_buffer(
            "attn_mask",
            torch.tril(torch.ones(self.max_len, self.max_len))[None, None, ...],
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # Guard identico al de agent/dt.py::DecisionTransformer._init_weights
        # -- necesario aca porque self.obs_encoder (ObservationSequenceEncoding)
        # ya carga y congela su propio encoder de pixeles ANTES de que este
        # self.initialize_weights() de nivel superior corra self.apply(...)
        # sobre todo el arbol (incluido obs_encoder.embed). Sin este guard,
        # pisaria los pesos pretrained ya cargados -- bug real encontrado en
        # esta sesion. Ver METODOLOGIA_DT_HDT.md, seccion 3.
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
                       episodio (si no se pasa, se asume 0..T-1). Se pasa
                       el mismo tensor a T_obs, T_act y al token de
                       return -- misma convención que ya usa dt.py.
        """
        batch_size, T = obs.shape[0], obs.shape[1]

        if timesteps is None:
            timesteps = (
                torch.arange(T, device=obs.device).unsqueeze(0).repeat(batch_size, 1)
            )

        if not self.pixel_obs:
            obs = (obs - self.obs_mean) / self.obs_std

        # T_obs / T_act: cada uno ya devuelve una secuencia contextualizada
        # y con posición propia, no hace falta sumarle time_emb de nuevo.
        s = self.obs_encoder(obs, timesteps=timesteps)
        a = self.action_encoder(action, timesteps=timesteps)
        r = self.return_embed(returns_to_go) + self.return_pos_embed(timesteps)

        x = (
            torch.stack([r, s, a], dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * T, self.n_embd)
        )

        for blk in self.blocks:
            x = blk(x, self.attn_mask)

        pred_a = self.action_head(x[:, 1::3])
        return pred_a


class HDTAgent(DTAgent):
    """
    DTAgent pero con HierarchicalDecisionTransformer en vez de
    DecisionTransformer. train/act/update_actor/update/
    compute_returns_to_go se heredan sin cambios: solo tocan atributos
    genéricos (self.model, self.opt, self.device, self.return_scale, ...)
    y self.model(...) tiene la misma firma en ambas clases.
    """

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

        self.model = HierarchicalDecisionTransformer(
            obs_shape, action_shape[0], self.config
        ).to(device)
        if obs_mean is not None:
            self.model.set_obs_stats(obs_mean, obs_std)
        if path is not None:
            # strict=False: ver nota equivalente en DTAgent.__init__ (dt.py)
            self.model.load_state_dict(payload["model"], strict=False)

        if use_adamw:
            self.opt, self.scheduler = configure_optimizer(
                self.model, lr, weight_decay, warmup_steps
            )
        else:
            self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.opt, lr_lambda=lambda step: 1.0
            )
        self.return_scale = self.config.return_scale
        print(
            "number of parameters: %e", sum(p.numel() for p in self.model.parameters())
        )
        self.train()
