from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    channels: int = 32
    time_points: int = 160
    sampling_rate: int = 128
    conditions: int = 3
    sessions: int = 2
    blocks: int = 6
    feature_dim: int = 80
    graph_layers: int = 4
    graph_neighbors: int = 8
    mamba_state_dim: int = 32
    mamba_expand: int = 2
    mamba_conv_width: int = 4
    dropout: float = 0.20
    sequence_length: int = 12
    sequence_stride: int = 6
    intra_trial_layers: int = 1
    inter_trial_layers: int = 2



SCALP_REGION = (
    0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 4, 4,
    4, 3, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7,
)



BANDS = ((4.0, 8.0), (8.0, 13.0), (13.0, 30.0))
