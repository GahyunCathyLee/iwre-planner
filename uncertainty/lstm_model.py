from __future__ import annotations

import torch
from torch import nn

from uncertainty.config import LSTMConfig


class LSTMUncertaintyEstimator(nn.Module):
    def __init__(self, config: LSTMConfig | None = None):
        super().__init__()
        self.config = config or LSTMConfig()
        if self.config.output_mode not in ("scalar", "gaussian"):
            raise ValueError(f"Unsupported output_mode: {self.config.output_mode}")
        self.encoder = nn.LSTM(
            input_size=self.config.input_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout if self.config.num_layers > 1 else 0.0,
            batch_first=True,
        )
        output_dim = 1 if self.config.output_mode == "scalar" else 2
        self.head = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.head_hidden_size),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.head_hidden_size, output_dim),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        _, (hidden, _) = self.encoder(x)
        raw = self.head(hidden[-1])
        if self.config.output_mode == "scalar":
            return {"sigma": torch.sigmoid(raw[:, 0])}
        mu = torch.sigmoid(raw[:, 0])
        logvar = torch.clamp(raw[:, 1], min=-10.0, max=5.0)
        return {"mu": mu, "logvar": logvar}
