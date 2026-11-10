from __future__ import annotations

import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import ModelConfig


class SelectiveMamba(nn.Module):
    def __init__(self, width: int, state_dim: int, conv_width: int, expand: int):
        super().__init__()
        inner = width * expand
        rank = math.ceil(width / 16)
        self.inner = inner
        self.state_dim = state_dim
        self.in_proj = nn.Linear(width, 2 * inner, bias=False)
        self.conv = nn.Conv1d(
            inner, inner, conv_width, groups=inner, padding=conv_width - 1
        )
        self.param_proj = nn.Linear(inner, rank + 2 * state_dim, bias=False)
        self.dt_proj = nn.Linear(rank, inner)
        self.out_proj = nn.Linear(inner, width, bias=False)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, state_dim + 1).float()).repeat(inner, 1)
        )
        self.D = nn.Parameter(torch.ones(inner))

        std = rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -std, std)
        dt = torch.exp(
            torch.rand(inner) * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        )
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def forward(self, sequence: Tensor) -> Tensor:
        length = sequence.shape[1]
        x, z = self.in_proj(sequence).chunk(2, dim=-1)
        x = F.silu(self.conv(x.transpose(1, 2))[..., :length]).transpose(1, 2)
        dt_rank, B, C = torch.split(
            self.param_proj(x), [self.dt_proj.in_features, self.state_dim, self.state_dim],
            dim=-1,
        )
        delta = F.softplus(self.dt_proj(dt_rank))

        # The state recurrence uses float32 for numerical stability.
        dtype = x.dtype
        x, z, delta, B, C = (t.float() for t in (x, z, delta, B, C))
        A = -torch.exp(self.A_log.float())
        D = self.D.float()
        state = torch.zeros(x.shape[0], self.inner, self.state_dim, device=x.device)
        outputs = []
        for t in range(length):
            dt_t = delta[:, t]
            state = (
                state * torch.exp(dt_t.unsqueeze(-1) * A)
                + x[:, t].unsqueeze(-1) * dt_t.unsqueeze(-1) * B[:, t].unsqueeze(1)
            )
            y = torch.einsum('bin,bn->bi', state, C[:, t]) + D * x[:, t]
            outputs.append(y * F.silu(z[:, t]))
        return self.out_proj(torch.stack(outputs, dim=1).to(dtype))



class BidirectionalMamba(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        width = config.feature_dim
        arguments = (width, config.mamba_state_dim, config.mamba_conv_width,
                     config.mamba_expand)
        self.norm = nn.LayerNorm(width)
        self.forward_core = SelectiveMamba(*arguments)
        self.backward_core = SelectiveMamba(*arguments)
        self.fusion = nn.Linear(2 * width, width, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        normalized = self.norm(x)
        left = self.forward_core(normalized)
        right = torch.flip(
            self.backward_core(torch.flip(normalized, dims=(1,))), dims=(1,)
        )
        return x + self.dropout(self.fusion(torch.cat((left, right), dim=-1)))
