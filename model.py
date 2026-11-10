from __future__ import annotations

import torch
from torch import Tensor, nn

from .config import ModelConfig
from .decoder import FactorizedComorbidityDecoder
from .encoder import TrialEncoder
from .mamba import BidirectionalMamba
from .pooling import AttentionPool


class FCTMamba(nn.Module):
    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__()
        self.config = config
        width = config.feature_dim
        self.trial_encoder = TrialEncoder(config)
        self.condition_embedding = nn.Embedding(config.conditions, width)
        self.session_embedding = nn.Embedding(config.sessions, width)
        self.block_embedding = nn.Embedding(config.blocks, width)
        self.transition_embedding = nn.Embedding(config.conditions ** 2, width)
        self.position_embedding = nn.Parameter(torch.zeros(1, config.sequence_length, width))
        nn.init.normal_(self.position_embedding, std=0.02)
        self.reliability_embedding = nn.Linear(1, width)
        self.across_trials = nn.Sequential(*[
            BidirectionalMamba(config) for _ in range(config.inter_trial_layers)
        ])
        self.trial_pool = AttentionPool(width)
        self.decoder = FactorizedComorbidityDecoder(config)

    def forward(
        self, eeg: Tensor, trial_type: Tensor, session: Tensor, block: Tensor
    ) -> dict[str, Tensor]:
        if eeg.ndim != 4 or eeg.shape[2:] != (self.config.channels, self.config.time_points):
            raise ValueError('eeg must have shape [batch, sequence, 32, 160]')
        batch, length, channels, time = eeg.shape
        if length != self.config.sequence_length:
            raise ValueError(f'sequence length must be {self.config.sequence_length}')
        for name, labels in [('trial_type', trial_type), ('session', session), ('block', block)]:
            if labels.shape != (batch, length):
                raise ValueError(f'{name} must have shape [batch, sequence]')

        feature, quality_logit = self.trial_encoder(
            eeg.reshape(batch * length, channels, time), trial_type.reshape(-1)
        )
        feature = feature.reshape(batch, length, -1)
        quality_logit = quality_logit.reshape(batch, length)
        reliability = torch.sigmoid(quality_logit)
        previous = torch.cat((trial_type[:, :1], trial_type[:, :-1]), dim=1)
        transition = previous * self.config.conditions + trial_type
        feature = (
            feature + self.condition_embedding(trial_type)
            + self.session_embedding(session) + self.block_embedding(block)
            + self.transition_embedding(transition)
            + self.position_embedding[:, :length]
            + self.reliability_embedding((reliability.detach() - 0.5).unsqueeze(-1))
        )
        feature = self.across_trials(feature)
        sequence_feature = self.trial_pool(feature, reliability.detach())
        result = self.decoder(sequence_feature)
        result['reliability_logit'] = quality_logit
        return result
