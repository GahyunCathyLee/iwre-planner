from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from noise_modeling.data.npz_dataset import NoiseNPZDataset
from noise_modeling.evaluation.metrics import compute_metrics
from noise_modeling.losses.gaussian_nll import gaussian_nll, mse_loss
from noise_modeling.models.factory import build_model
from noise_modeling.utils.config import load_config, save_json
from noise_modeling.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit-shards", type=int, default=None)
    parser.add_argument("--output-dir", default=None, help="Override config output_dir.")
    return parser.parse_args()


def make_dataset(config: Dict, split: str, limit_shards: int | None = None) -> NoiseNPZDataset:
    data_cfg = config["data"]
    return NoiseNPZDataset(
        data_root=data_cfg["npz_root"],
        split=split,
        feature_set=data_cfg["feature_set"],
        normalization_path=data_cfg["normalization_stats"],
        use_sequence=bool(data_cfg.get("use_sequence", False)),
        normalize_features=bool(data_cfg.get("normalize_features", True)),
        normalize_targets=bool(data_cfg.get("normalize_targets", True)),
        limit_shards=limit_shards,
    )


def compute_loss(
    config: Dict,
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    target_std: torch.Tensor,
    normalize_targets: bool,
) -> torch.Tensor:
    loss_name = config["training"].get("loss", "mse")
    target = batch["target"]
    if loss_name == "gaussian_nll":
        loss = gaussian_nll(target, outputs["mu"], outputs["sigma"], full=True)
    elif loss_name == "mse":
        loss = mse_loss(target, outputs["mu"])
    else:
        raise ValueError(f"Unknown loss: {loss_name}")

    lambda_state = float(config["training"].get("lambda_state", 0.0))
    if lambda_state > 0.0:
        mu_state = outputs["mu"] * target_std if normalize_targets else outputs["mu"]
        corrected = batch["obs_state"] - mu_state
        loss = loss + lambda_state * mse_loss(batch["true_state"], corrected)
    return loss


@torch.no_grad()
def evaluate_model(model, loader, device, target_std: torch.Tensor, normalize_targets: bool) -> Tuple[float, Dict[str, float]]:
    model.eval()
    losses = []
    pred_mu = []
    pred_sigma = []
    targets = []
    obs_states = []
    true_states = []
    target_std = target_std.to(device)
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(batch["features"])
        if "sigma" in outputs:
            loss = gaussian_nll(batch["target"], outputs["mu"], outputs["sigma"], full=True)
        else:
            loss = mse_loss(batch["target"], outputs["mu"])
        losses.append(float(loss.item()))

        mu = outputs["mu"]
        sigma = outputs.get("sigma")
        target = batch["target"]
        if normalize_targets:
            mu = mu * target_std
            target = target * target_std
            if sigma is not None:
                sigma = sigma * target_std
        pred_mu.append(mu.cpu().numpy())
        if sigma is not None:
            pred_sigma.append(sigma.cpu().numpy())
        targets.append(target.cpu().numpy())
        obs_states.append(batch["obs_state"].cpu().numpy())
        true_states.append(batch["true_state"].cpu().numpy())

    mu_np = np.concatenate(pred_mu, axis=0)
    sigma_np = np.concatenate(pred_sigma, axis=0) if pred_sigma else None
    metrics = compute_metrics(
        target=np.concatenate(targets, axis=0),
        mu=mu_np,
        sigma=sigma_np,
        obs_state=np.concatenate(obs_states, axis=0),
        true_state=np.concatenate(true_states, axis=0),
    )
    return float(np.mean(losses)), metrics


def append_metrics(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    set_seed(int(config.get("seed", 42)))
    device = torch.device(args.device)

    train_dataset = make_dataset(config, "train", args.limit_shards)
    val_dataset = make_dataset(config, "val", args.limit_shards)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"].get("batch_size", 512)),
        shuffle=True,
        num_workers=int(config["training"].get("num_workers", 2)),
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["training"].get("eval_batch_size", config["training"].get("batch_size", 512))),
        shuffle=False,
        num_workers=int(config["training"].get("num_workers", 2)),
        pin_memory=device.type == "cuda",
    )

    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"].get("lr", 1.0e-3)),
        weight_decay=float(config["training"].get("weight_decay", 1.0e-5)),
    )

    output_dir = Path(config["output_dir"]) / config["experiment_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "config.json", config)
    print(f"[INFO] checkpoints: {output_dir}")

    best_val = float("inf")
    patience = int(config["training"].get("early_stopping_patience", 15))
    stale_epochs = 0
    target_std = train_dataset.target_std
    normalize_targets = bool(config["data"].get("normalize_targets", True))

    for epoch in range(1, int(config["training"].get("epochs", 100)) + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch["features"])
            loss = compute_loss(config, outputs, batch, target_std.to(device), normalize_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"].get("grad_clip", 5.0)))
            optimizer.step()
            train_losses.append(float(loss.item()))

        val_loss, val_metrics = evaluate_model(model, val_loader, device, target_std, normalize_targets)
        train_loss = float(np.mean(train_losses))
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **val_metrics}
        append_metrics(output_dir / "history.csv", row)
        print(f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f} noise_rmse={val_metrics['noise_rmse']:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "target_std": target_std.tolist(),
                    "best_val_loss": best_val,
                    "epoch": epoch,
                },
                output_dir / "best.pt",
            )
            print(f"[INFO] saved best checkpoint: {output_dir / 'best.pt'}")
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"early stopping at epoch {epoch}")
                break


if __name__ == "__main__":
    main()
