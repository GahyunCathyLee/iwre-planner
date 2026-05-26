from __future__ import annotations

from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F

from noise_modeling.models.mlp import build_mlp


class EgoCentricHistoryNoiseModelV3(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        head_hidden_dims: Iterable[int] = (128, 64),
        dropout: float = 0.0,
        min_sigma: float = 1.0e-4,
        max_sigma: float | None = None,
        output_type: str = "heteroscedastic",
    ):
        super().__init__()
        self.encoder = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = 8 if output_type == "heteroscedastic" else 4
        self.head = build_mlp(hidden_dim, head_hidden_dims, out_dim, dropout)
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.output_type = output_type

    def forward(self, features: torch.Tensor):
        _, hidden = self.encoder(features)
        last = hidden[-1]
        output = self.head(last)
        if self.output_type == "mean":
            return {"mu": output}
        mu = output[:, :4]
        sigma = F.softplus(output[:, 4:]) + self.min_sigma
        if self.max_sigma is not None:
            sigma = torch.clamp(sigma, max=self.max_sigma)
        return {"mu": mu, "sigma": sigma}

