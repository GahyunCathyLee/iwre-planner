from __future__ import annotations

import torch


def gaussian_nll_loss(mu: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    logvar = torch.clamp(logvar, min=-10.0, max=5.0)
    return 0.5 * (torch.exp(-logvar) * (target - mu) ** 2 + logvar).mean()
