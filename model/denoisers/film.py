import math
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    """Classic sinusoidal embedding used for diffusion timesteps."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        device = t.device
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class FiLMHead(nn.Module):
    """Produce FiLM scale and shift from timestep embedding."""

    def __init__(self, time_dim: int, channels: int, hidden_dim=None) -> None:
        super().__init__()
        hidden = hidden_dim or time_dim
        self.net = nn.Sequential(
            nn.Linear(time_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels * 2)
        )
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, time_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gamma_beta = self.net(time_emb)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return gamma, beta


def apply_film(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    while gamma.dim() < x.dim():
        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)
    return x * (gamma + 1.0) + beta
