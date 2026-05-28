from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from interaction_modeling.models.grip_alpha import GRIPFuturePredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create neighbor occlusion alpha teacher from a trained GRIP future predictor.")
    parser.add_argument("--data-root", default="outputs/interaction_grip")
    parser.add_argument("--checkpoint", default="outputs/interaction_grip_future/best.pt")
    parser.add_argument("--out-root", default="outputs/interaction_grip_teacher")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--temperature", type=float, default=1.0)
    return parser.parse_args()


@torch.no_grad()
def compute_teacher(model: GRIPFuturePredictor, x: torch.Tensor, adj: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    full_pred = model(x, adj)
    full_err = F.smooth_l1_loss(full_pred, target, reduction="none").mean(dim=(1, 2))
    deltas = []
    for node in range(1, 9):
        x_masked = x.clone()
        adj_masked = adj.clone()
        x_masked[:, :, :, node] = 0.0
        adj_masked[:, node, :] = 0.0
        adj_masked[:, :, node] = 0.0
        adj_masked[:, node, node] = 1.0
        pred = model(x_masked, adj_masked)
        err = F.smooth_l1_loss(pred, target, reduction="none").mean(dim=(1, 2))
        deltas.append((err - full_err).clamp_min(0.0))
    delta = torch.stack(deltas, dim=1) * valid_mask[:, 1:]
    max_delta = delta.max(dim=1, keepdim=True).values.clamp_min(1.0e-6)
    return (delta / max_delta).clamp(0.0, 1.0)


def process_shard(model: GRIPFuturePredictor, in_path: Path, out_path: Path, device: torch.device) -> None:
    with np.load(in_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    x = torch.from_numpy(arrays["x"].astype(np.float32)).to(device)
    adj = torch.from_numpy(arrays["adj"].astype(np.float32)).to(device)
    target = torch.from_numpy(arrays["target"].astype(np.float32)).to(device)
    valid_mask = torch.from_numpy(arrays["valid_mask"].astype(np.float32)).to(device)
    teacher = compute_teacher(model, x, adj, target, valid_mask).cpu().numpy().astype(np.float32)
    arrays["alpha_teacher"] = teacher
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = GRIPFuturePredictor(
        in_channels=int(ckpt["in_channels"]),
        future_frames=int(ckpt["future_frames"]),
        hidden_channels=int(ckpt.get("hidden_channels", 64)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    for split in ("train", "val", "test"):
        in_paths = sorted((Path(args.data_root) / split).glob("*.npz"))
        for in_path in in_paths:
            out_path = Path(args.out_root) / split / in_path.name
            process_shard(model, in_path, out_path, device)
            print(f"[INFO] wrote {out_path}")


if __name__ == "__main__":
    main()

