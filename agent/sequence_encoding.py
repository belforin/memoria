import torch
import torch.nn as nn

import utils
from agent.modules.attention import Block


class SequenceEncoding(nn.Module):
    """
    Clase base compartida por ObservationSequenceEncoding y
    ActionSequenceEncoding.

    Sigue el mismo patrón que DecisionTransformer (dt.py):
    - proyección lineal de la modalidad a n_embd
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

    def __init__(self, input_dim, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.traj_length = config.traj_length
        self.episode_length = config.episode_length
        # un solo token por timestep (no *2 ni *3 como en GPT/DT)
        self.max_len = config.traj_length

        self.embed = nn.Linear(input_dim, self.n_embd)
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
        x:         (B, T, input_dim)
        timesteps: (B, T) long tensor con el índice real dentro del
                   episodio (si no se pasa, se asume 0..T-1)

        return: (B, T, n_embd) -- secuencia codificada y contextualizada
        """
        batch_size, T, _ = x.size()

        if timesteps is None:
            timesteps = (
                torch.arange(T, device=x.device).unsqueeze(0).repeat(batch_size, 1)
            )
        time_emb = self.pos_embed(timesteps)  # (B, T, n_embd)

        x = self.embed(x) + time_emb

        for blk in self.blocks:
            x = blk(x, self.attn_mask)

        return x


class ObservationSequenceEncoding(SequenceEncoding):
    def __init__(self, obs_dim, config):
        super().__init__(obs_dim, config)


class ActionSequenceEncoding(SequenceEncoding):
    def __init__(self, action_dim, config):
        super().__init__(action_dim, config)
