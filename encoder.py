from __future__ import annotations

import torch
from torch import Tensor, nn

from .config import BANDS, SCALP_REGION, ModelConfig
from .graph import ConditionAwareSignedGraph, SignedGraphLayer
from .mamba import BidirectionalMamba
from .pooling import AttentionPool


class TrialEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.sampling_rate = config.sampling_rate
        self.register_buffer('region', torch.tensor(SCALP_REGION), persistent=False)
        self.waveform_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, 8, kernel_size=k, padding=k // 2, bias=False),
                nn.BatchNorm1d(8), nn.GELU(),
            )
            for k in (7, 15, 31)
        ])
        width = config.feature_dim
        self.node_projection = nn.Sequential(
            nn.Linear(28, width), nn.LayerNorm(width), nn.GELU()
        )
        self.graph_builder = ConditionAwareSignedGraph(config)
        self.graph_layers = nn.ModuleList([
            SignedGraphLayer(config) for _ in range(config.graph_layers)
        ])
        self.graph_pool = AttentionPool(width)
        self.region_conv = nn.Sequential(
            nn.Conv1d(8, width, kernel_size=9, stride=4, padding=4, bias=False),
            nn.BatchNorm1d(width), nn.GELU(),
        )
        self.within_trial = nn.Sequential(*[
            BidirectionalMamba(config) for _ in range(config.intra_trial_layers)
        ])
        self.temporal_pool = AttentionPool(width)
        self.fuse = nn.Sequential(
            nn.Linear(width * 2, width), nn.LayerNorm(width),
            nn.GELU(), nn.Dropout(config.dropout),
        )
        self.reliability_head = nn.Sequential(
            nn.Linear(width * 2 + 3, width // 2), nn.GELU(),
            nn.Dropout(config.dropout), nn.Linear(width // 2, 1),
        )

    def _spectral_features(self, x: Tensor) -> Tensor:
        spectrum = torch.fft.rfft(x.float(), dim=-1)
        power = spectrum.abs().square()
        freq = torch.fft.rfftfreq(x.shape[-1], 1 / self.sampling_rate).to(x.device)
        features = [
            torch.log1p(power[..., (freq >= lo) & (freq < hi)].mean(dim=-1))
            for lo, hi in BANDS
        ]
        theta_phase = torch.angle(spectrum[..., (freq - 6).abs().argmin()])
        beta_amplitude = power[..., (freq >= 18) & (freq < 26)].mean(dim=-1).sqrt()
        features.append(torch.sin(theta_phase) * torch.log1p(beta_amplitude))
        return torch.stack(features, dim=-1)

    def forward(self, eeg: Tensor, condition: Tensor) -> tuple[Tensor, Tensor]:
        count, channels, time = eeg.shape
        signal = eeg.reshape(count * channels, 1, time)
        waveform = torch.cat(
            [branch(signal).mean(dim=-1) for branch in self.waveform_branches], dim=-1
        ).reshape(count, channels, -1)
        nodes = self.node_projection(
            torch.cat((waveform, self._spectral_features(eeg)), dim=-1)
        )
        positive, negative = self.graph_builder(eeg, condition)
        for layer in self.graph_layers:
            nodes = layer(nodes, positive, negative)
        graph_feature = self.graph_pool(nodes)

        regions = torch.stack([
            eeg[:, self.region == region].mean(dim=1) for region in range(8)
        ], dim=1)
        tokens = self.within_trial(self.region_conv(regions).transpose(1, 2))
        temporal_feature = self.temporal_pool(tokens)
        merged = torch.cat((graph_feature, temporal_feature), dim=-1)
        feature = self.fuse(merged)
        stats = torch.stack((
            eeg.std(dim=(1, 2)),
            eeg.diff(dim=-1).abs().mean(dim=(1, 2)),
            (eeg.abs() > 3.5).float().mean(dim=(1, 2)),
        ), dim=-1)
        reliability_logit = self.reliability_head(
            torch.cat((merged, stats), dim=-1)
        ).squeeze(-1)
        return feature, reliability_logit
