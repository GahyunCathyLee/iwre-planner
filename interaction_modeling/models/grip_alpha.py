from __future__ import annotations

import torch
import torch.nn as nn

from interaction_modeling.models.stgcn import GRIPEncoder


class GRIPFuturePredictor(nn.Module):
    """GRIP-style encoder with an ego future trajectory head."""

    def __init__(self, in_channels: int, future_frames: int, hidden_channels: int = 64, max_hop: int = 2):
        super().__init__()
        self.future_frames = future_frames
        self.encoder = GRIPEncoder(in_channels, hidden_channels, max_hop)
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_channels, future_frames * 2),
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x, adj)
        ego_h = h[:, :, -1, 0]
        return self.head(ego_h).view(-1, self.future_frames, 2)


class GRIPAlphaEstimator(nn.Module):
    """GRIP-style encoder with neighbor-wise alpha head."""

    def __init__(self, in_channels: int, hidden_channels: int = 64, max_hop: int = 2):
        super().__init__()
        self.encoder = GRIPEncoder(in_channels, hidden_channels, max_hop)
        self.alpha_head = nn.Sequential(
            nn.Conv1d(hidden_channels * 2, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=False),
            nn.Conv1d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.encoder(x, adj)
        node_h = h[:, :, -1, :]
        ego_h = node_h[:, :, 0:1].expand(-1, -1, node_h.size(-1) - 1)
        nbr_h = node_h[:, :, 1:]
        logits = self.alpha_head(torch.cat([ego_h, nbr_h], dim=1)).squeeze(1)
        alpha = torch.sigmoid(logits)
        if valid_mask is not None:
            alpha = alpha * valid_mask[:, 1:]
        return alpha

