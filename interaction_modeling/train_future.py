from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from interaction_modeling.data.grip_dataset import InteractionGRIPDataset
from interaction_modeling.models.grip_alpha import GRIPFuturePredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="outputs/interaction_grip")
    parser.add_argument("--output-dir", default="outputs/interaction_grip_future")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--limit-shards", type=int, default=None)
    return parser.parse_args()


def run_epoch(model, loader, device, optimizer=None) -> float:
    model.train(optimizer is not None)
    losses = []
    for batch in loader:
        x = batch["x"].to(device)
        adj = batch["adj"].to(device)
        target = batch["target"].to(device)
        pred = model(x, adj)
        loss = F.smooth_l1_loss(pred, target)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses))


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    train_ds = InteractionGRIPDataset(args.data_root, "train", args.limit_shards)
    val_ds = InteractionGRIPDataset(args.data_root, "val", args.limit_shards)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    in_channels = int(train_ds.x.shape[1])
    future_frames = int(train_ds.target.shape[1])
    model = GRIPFuturePredictor(in_channels, future_frames, args.hidden_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.0e-5)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, device, optimizer)
        with torch.no_grad():
            val_loss = run_epoch(model, val_loader, device)
        print(f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        if val_loss < best:
            best = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "in_channels": in_channels,
                    "future_frames": future_frames,
                    "hidden_channels": args.hidden_channels,
                    "best_val_loss": best,
                },
                out_dir / "best.pt",
            )
            print(f"[INFO] saved {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()

