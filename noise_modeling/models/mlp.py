from __future__ import annotations

from typing import Iterable, List

import torch
from torch import nn
import torch.nn.functional as F


def build_mlp(input_dim: int, hidden_dims: Iterable[int], output_dim: int, dropout: float = 0.0) -> nn.Sequential:
    layers: List[nn.Module] = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(nn.ReLU())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class EgoCentricMLPNoiseRegressorV1(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Iterable[int], dropout: float = 0.0):
        super().__init__()
        self.net = build_mlp(input_dim, hidden_dims, 4, dropout)
        self.output_type = "mean"

    def forward(self, features: torch.Tensor):
        return {"mu": self.net(features)}


class EgoCentricHeteroscedasticMLPNoiseModelV2(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int],
        dropout: float = 0.0,
        min_sigma: float = 1.0e-4,
        max_sigma: float | None = None,
    ):
        super().__init__()
        self.net = build_mlp(input_dim, hidden_dims, 8, dropout)
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.output_type = "heteroscedastic"

    def forward(self, features: torch.Tensor):
        output = self.net(features)
        mu = output[:, :4]
        sigma = F.softplus(output[:, 4:]) + self.min_sigma
        if self.max_sigma is not None:
            sigma = torch.clamp(sigma, max=self.max_sigma)
        return {"mu": mu, "sigma": sigma}

