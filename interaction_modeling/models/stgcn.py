from __future__ import annotations

import torch
import torch.nn as nn


class ConvTemporalGraphical(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels * kernel_size, kernel_size=(1, 1))

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        if adj.size(1) != self.kernel_size:
            raise ValueError(f"Expected {self.kernel_size} adjacency hops, got {adj.size(1)}")
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        return torch.einsum("nkctv,nkvw->nctw", x, adj).contiguous()


class STGCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, spatial_kernel: int, temporal_kernel: int = 5):
        super().__init__()
        padding = ((temporal_kernel - 1) // 2, 0)
        self.gcn = ConvTemporalGraphical(in_channels, out_channels, spatial_kernel)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=(temporal_kernel, 1), padding=padding),
            nn.BatchNorm2d(out_channels),
        )
        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1), nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return self.relu(self.tcn(self.gcn(x, adj)) + self.residual(x))


def normalize_adjacency(adj: torch.Tensor, max_hop: int = 2) -> torch.Tensor:
    """Return stacked [I, A, A^2] normalized adjacency for a batch of raw adjacency."""
    if adj.ndim != 3:
        raise ValueError("adj must have shape (N, V, V)")
    n, v, _ = adj.shape
    eye = torch.eye(v, dtype=adj.dtype, device=adj.device).expand(n, v, v)
    hops = [eye]
    if max_hop >= 1:
        hops.append((adj > 0).to(adj.dtype))
    for _ in range(2, max_hop + 1):
        hops.append((torch.matmul(hops[-1], hops[1]) > 0).to(adj.dtype))
    stacked = torch.stack(hops[: max_hop + 1], dim=1)
    degree = stacked.sum(dim=-2, keepdim=True).clamp_min(1.0)
    return stacked / degree


class GRIPEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, max_hop: int = 2):
        super().__init__()
        self.max_hop = max_hop
        spatial_kernel = max_hop + 1
        self.input_bn = nn.BatchNorm2d(in_channels)
        self.blocks = nn.ModuleList(
            [
                STGCNBlock(in_channels, hidden_channels, spatial_kernel),
                STGCNBlock(hidden_channels, hidden_channels, spatial_kernel),
                STGCNBlock(hidden_channels, hidden_channels, spatial_kernel),
            ]
        )

    def forward(self, x: torch.Tensor, adj_raw: torch.Tensor) -> torch.Tensor:
        adj = normalize_adjacency(adj_raw, self.max_hop)
        h = self.input_bn(x)
        for block in self.blocks:
            h = block(h, adj)
        return h

