from __future__ import annotations

import torch
from torch import Tensor, nn


class AttentionPool(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width // 2),
            nn.Tanh(), nn.Linear(width // 2, 1),
        )

    def forward(self, x: Tensor, reliability: Tensor | None = None) -> Tensor:
        scores = self.score(x).squeeze(-1)
        if reliability is not None:
            scores = scores + torch.log(reliability.clamp(0.08, 1.0))
        weights = torch.softmax(scores, dim=1)
        return (x * weights.unsqueeze(-1)).sum(dim=1)
