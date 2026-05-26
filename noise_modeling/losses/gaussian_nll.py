from __future__ import annotations

import math

import torch


def gaussian_nll(target: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, full: bool = True) -> torch.Tensor:
    var = sigma.square()
    loss = 0.5 * ((target - mu).square() / var + torch.log(var))
    if full:
        loss = loss + 0.5 * math.log(2.0 * math.pi)
    return loss.mean()


def mse_loss(target: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    return (target - mu).square().mean()

