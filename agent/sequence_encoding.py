import torch
import torch.nn as nn

import utils
from agent.modules.attention import Block
from agent.modules.pixel_encoder import PixelEncoder
from agent.modules.load_pretrained_encoder import load_procgen_impala


class SequenceEncoding(nn.Module):
    """
    Clase base compartida por ObservationSequenceEncoding y
    ActionSequenceEncoding.

    Sigue el mismo patrón que DecisionTransformer (dt.py):
    - proyección de la modalidad a n_embd (lineal, o un encoder de pixeles/
      un nn.Embedding discreto via `embed_module` -- ver subclases)
    - positional embedding aprendido, indexado por timestep real del
      episodio (no por posición dentro del batch), igual que en DT
    - bloques de self-attention causal

    A diferencia de DT, acá cada modalidad se codifica por separado
    (un token por timestep, no intercalado con otras modalidades), así
    que el positional embedding queda bien definido incluso si después
    se usa esta secuencia de forma independiente.

    No incluye head de predicción: esto es solo el encoder. Si hace
    falta una salida (reconstrucción, predicción, etc.) se agrega en
    la clase que use este encoder.
    """

    def __init__(self, input_dim, config, embed_module=None, discrete=False):
        super().__init__()
        self.n_embd = config.n_embd
        self.traj_length = config.traj_length
        self.episode_length = config.episode_length
        # un solo token por timestep (no *2 ni *3 como en GPT/DT)
        self.max_len = config.traj_length
        # acción discreta: el squeeze/long antes de embed()/nn.Embedding se
        # hace en forward() (ver ahí). Sin efecto sobre ObservationSequenceEncoding
        # (siempre discrete=False).
        self.discrete = discrete

        self.embed = embed_module if embed_module is not None else nn.Linear(input_dim, self.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        self.initialize_weights()

    def initialize_weights(self):
        self.pos_embed = nn.Embedding(self.episode_length, self.n_embd)
        self.register_buffer(
            "attn_mask",
            torch.tril(torch.ones(self.max_len, self.max_len))[None, None, ...],
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # Guard identico al de agent/dt.py::DecisionTransformer._init_weights
        # (no reinicializar pesos ya congelados/pretrained).
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

    def forward(self, x, timesteps=None):
        """
        x:         (B, T, input_dim) obs/accion vectorial, (B, T, H, W, C)
                   uint8 obs de pixeles, o (B, T, 1)/(B, T) int64 indices de
                   accion (self.discrete)
        timesteps: (B, T) long tensor con el índice real dentro del
                   episodio (si no se pasa, se asume 0..T-1)

        return: (B, T, n_embd) -- secuencia codificada y contextualizada
        """
        batch_size, T = x.shape[0], x.shape[1]

        if timesteps is None:
            timesteps = (
                torch.arange(T, device=x.device).unsqueeze(0).repeat(batch_size, 1)
            )
        time_emb = self.pos_embed(timesteps)  # (B, T, n_embd)

        if self.discrete:
            # (B, T, 1) o (B, T) int -> (B, T) long, mismo patron que
            # agent/dt.py::DecisionTransformer.forward y agent/mdp.py de
            # Benjamin Mancilla (rama upstream/hier-procgen).
            x = x.long()
            if x.dim() == 3 and x.size(-1) == 1:
                x = x.squeeze(-1)

        x = self.embed(x) + time_emb

        for blk in self.blocks:
            x = blk(x, self.attn_mask)

        return x


class ObservationSequenceEncoding(SequenceEncoding):
    def __init__(self, obs_shape, config):
        """
        obs_shape: int (o tupla de 1) = obs vectorial (D4RL, Etapa 1);
        tupla de 3 (H,W,C) = obs de pixeles (CoinRun, Etapa 2). Ver
        METODOLOGIA_DT_HDT.md, seccion 3.
        """
        if isinstance(obs_shape, int):
            obs_shape = (obs_shape,)
        pixel_obs = len(obs_shape) == 3

        if pixel_obs:
            pixel_encoder_type = str(getattr(config, "pixel_encoder_type", "procgen_impala"))
            embed_module = PixelEncoder(obs_shape, config.n_embd, encoder_type=pixel_encoder_type)
            input_dim = None
        else:
            embed_module = None
            input_dim = obs_shape[0]

        super().__init__(input_dim, config, embed_module=embed_module)
        self.pixel_obs = pixel_obs

        if pixel_obs:
            # Cargar los pesos pretrained DESPUES de self.initialize_weights()
            # (ya corrido dentro de super().__init__): ver la nota equivalente
            # en agent/dt.py::DecisionTransformer.__init__ (el orden importa,
            # bug encontrado en agent/mdp.py de Benjamin al leer su codigo).
            ckpt_path = getattr(config, "pretrained_encoder_path", None)
            if ckpt_path is not None:
                load_procgen_impala(self.embed, ckpt_path, freeze=True)


class ActionSequenceEncoding(SequenceEncoding):
    def __init__(self, action_dim, config):
        discrete = bool(getattr(config, "discrete_actions", False))
        if discrete:
            embed_module = nn.Embedding(action_dim, config.n_embd)
            input_dim = None
        else:
            embed_module = None
            input_dim = action_dim
        super().__init__(input_dim, config, embed_module=embed_module, discrete=discrete)
