from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import BANDS, SCALP_REGION, ModelConfig


class ConditionAwareSignedGraph(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.sampling_rate = config.sampling_rate
        self.neighbors = config.graph_neighbors
        self.condition_embedding = nn.Embedding(config.conditions, 16)
        self.band_gate = nn.Sequential(
            nn.Linear(22, 24), nn.GELU(), nn.Dropout(config.dropout / 2),
            nn.Linear(24, 3),
        )
        self.prior_gate = nn.Sequential(
            nn.Linear(19, 16), nn.GELU(), nn.Linear(16, 1), nn.Sigmoid()
        )
        region = torch.tensor(SCALP_REGION)
        distances = (region[:, None] - region[None, :]).abs().float()
        prior = torch.exp(-0.75 * distances)
        prior.fill_diagonal_(0)
        weights, indices = prior.topk(self.neighbors, dim=-1)
        prior = torch.zeros_like(prior).scatter(-1, indices, weights)
        self.register_buffer('regional_prior', prior / prior.sum(-1, keepdim=True).clamp_min(1e-6))

    @staticmethod
    def _correlation(x: Tensor) -> Tensor:
        x = x - x.mean(-1, keepdim=True)
        x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-5)
        return torch.bmm(x, x.transpose(1, 2))

    def forward(self, eeg: Tensor, condition: Tensor) -> tuple[Tensor, Tensor]:
        eeg = eeg.float()
        frequencies = torch.fft.rfftfreq(
            eeg.shape[-1], d=1.0 / self.sampling_rate
        ).to(eeg.device)
        spectrum = torch.fft.rfft(eeg, dim=-1)
        signals = [
            torch.fft.irfft(
                spectrum * ((frequencies >= lo) & (frequencies < hi)),
                n=eeg.shape[-1], dim=-1,
            )
            for lo, hi in BANDS
        ]
        correlations = torch.stack([self._correlation(s) for s in signals], dim=1)
        power = torch.stack(
            [torch.log1p(s.square().mean(dim=(1, 2))) for s in signals], dim=-1
        )
        strength = correlations.abs().mean(dim=(-1, -2))
        context = self.condition_embedding(condition)
        band_weight = torch.softmax(
            self.band_gate(torch.cat((context, power, strength), dim=-1)), dim=-1
        )
        adjacency = (band_weight[:, :, None, None] * correlations).sum(dim=1)
        diagonal = torch.eye(adjacency.shape[-1], device=adjacency.device, dtype=torch.bool)
        adjacency = adjacency.masked_fill(diagonal.unsqueeze(0), 0.0)

        positive = adjacency.clamp_min(0)
        negative = (-adjacency).clamp_min(0)
        prior_weight = 0.12 + 0.38 * self.prior_gate(torch.cat((context, power), dim=-1))
        positive = positive + prior_weight.unsqueeze(-1) * self.regional_prior

        def sparse_normalize(edges: Tensor) -> Tensor:
            values, indices = edges.topk(self.neighbors, dim=-1)
            sparse = torch.zeros_like(edges).scatter(-1, indices, values)
            return sparse / sparse.sum(dim=-1, keepdim=True).clamp_min(1e-5)

        return sparse_normalize(positive), sparse_normalize(negative)



class SignedGraphLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        width = config.feature_dim
        self.self_projection = nn.Linear(width, width)
        self.positive_projection = nn.Linear(width, width, bias=False)
        self.negative_projection = nn.Linear(width, width, bias=False)
        self.gate = nn.Sequential(
            nn.Linear(width * 3, width), nn.GELU(),
            nn.Linear(width, width), nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, nodes: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
        base = self.self_projection(nodes)
        pos = self.positive_projection(torch.bmm(positive, nodes))
        neg = self.negative_projection(torch.bmm(negative, nodes))
        gate = self.gate(torch.cat((base, pos, neg), dim=-1))
        return self.norm(nodes + self.dropout(F.gelu(base + gate * pos - (1 - gate) * neg)))
