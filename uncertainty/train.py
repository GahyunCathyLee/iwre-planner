from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from uncertainty.config import LSTMConfig, TrainConfig
from uncertainty.dataset import UncertaintyCSVDataset, discover_neighbor_logs, split_paths
from uncertainty.losses import gaussian_nll_loss
from uncertainty.lstm_model import LSTMUncertaintyEstimator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Track B LSTM uncertainty estimator from B1 neighbor CSV logs.")
    parser.add_argument("--log-root", default="outputs/logs")
    parser.add_argument("--pattern", default="*_B1*_nbr.csv")
    parser.add_argument("--output-dir", default="outputs/uncertainty/checkpoints/track_b_lstm")
    parser.add_argument("--history-length", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-mode", choices=["scalar", "gaussian"], default="gaussian")
    parser.add_argument("--limit-files", type=int, default=None)
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, output_mode: str) -> float:
    model.eval()
    losses = []
    for batch in loader:
        features = batch["features"].to(device)
        target = batch["target"].to(device)
        output = model(features)
        if output_mode == "gaussian":
            loss = gaussian_nll_loss(output["mu"], output["logvar"], target)
        else:
            loss = F.mse_loss(output["sigma"], target)
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("inf")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    paths = discover_neighbor_logs(args.log_root, args.pattern)
    if args.limit_files is not None:
        paths = paths[: args.limit_files]
    if not paths:
        raise FileNotFoundError(f"No neighbor logs found under {args.log_root} with pattern {args.pattern}")
    train_paths, val_paths = split_paths(paths, args.val_fraction, args.seed)
    if not val_paths:
        val_paths = train_paths

    train_dataset = UncertaintyCSVDataset(train_paths, history_length=args.history_length)
    val_dataset = UncertaintyCSVDataset(
        val_paths,
        history_length=args.history_length,
        feature_mean=train_dataset.feature_mean,
        feature_std=train_dataset.feature_std,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    lstm_config = LSTMConfig(output_mode=args.output_mode)
    train_config = TrainConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
    )
    device = torch.device(args.device)
    model = LSTMUncertaintyEstimator(lstm_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            features = batch["features"].to(device)
            target = batch["target"].to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(features)
            if args.output_mode == "gaussian":
                loss = gaussian_nll_loss(output["mu"], output["logvar"], target)
            else:
                loss = F.mse_loss(output["sigma"], target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_losses.append(float(loss.item()))
        val_loss = evaluate(model, val_loader, device, args.output_mode)
        scheduler.step(val_loss)
        print(f"epoch={epoch:03d} train_loss={np.mean(train_losses):.6f} val_loss={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            stale = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "lstm_config": asdict(lstm_config),
                    "train_config": asdict(train_config),
                    "feature_mean": train_dataset.feature_mean.tolist(),
                    "feature_std": train_dataset.feature_std.tolist(),
                    "history_length": args.history_length,
                    "best_val_loss": best_val,
                    "epoch": epoch,
                },
                output_dir / "best.pt",
            )
            with (output_dir / "metadata.json").open("w") as fh:
                json.dump({"train_files": [str(p) for p in train_paths], "val_files": [str(p) for p in val_paths]}, fh, indent=2)
        else:
            stale += 1
            if stale >= TrainConfig().early_stop_patience:
                print(f"early stopping at epoch {epoch}")
                break


if __name__ == "__main__":
    main()
