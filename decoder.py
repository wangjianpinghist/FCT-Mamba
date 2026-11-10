from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import ModelConfig


class FactorizedComorbidityDecoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        width = config.feature_dim
        self.ts_head = nn.Linear(width, 1)
        self.adhd_head = nn.Linear(width, 1)
        self.interaction_head = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width // 2),
            nn.GELU(), nn.Dropout(config.dropout), nn.Linear(width // 2, 1),
        )
        self.residual_head = nn.Linear(width, 4)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        ts = self.ts_head(x).squeeze(-1)
        adhd = self.adhd_head(x).squeeze(-1)
        base = torch.stack((
            F.logsigmoid(-ts) + F.logsigmoid(-adhd),
            F.logsigmoid(-ts) + F.logsigmoid(adhd),
            F.logsigmoid(ts) + F.logsigmoid(-adhd),
            F.logsigmoid(ts) + F.logsigmoid(adhd),
        ), dim=-1)
        interaction_logit = self.interaction_head(x).squeeze(-1)
        interaction = torch.tanh(interaction_logit)
        direction = base.new_tensor((-0.25, -0.35, -0.35, 0.95))
        correction = 0.42 * interaction.unsqueeze(-1) * direction
        residual = 0.12 * torch.tanh(self.residual_head(x))
        return {
            'logits': base + correction + residual,
            'ts_logit': ts,
            'adhd_logit': adhd,
            'interaction_logit': interaction_logit,
        }
