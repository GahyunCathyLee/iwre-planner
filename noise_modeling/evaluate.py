from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from noise_modeling.models.factory import build_model
from noise_modeling.train import evaluate_model, make_dataset
from noise_modeling.utils.config import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit-shards", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    config = checkpoint["config"]
    device = torch.device(args.device)
    dataset = make_dataset(config, args.split, args.limit_shards)
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"].get("eval_batch_size", config["training"].get("batch_size", 512))),
        shuffle=False,
        num_workers=int(config["training"].get("num_workers", 2)),
        pin_memory=device.type == "cuda",
    )
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    loss, metrics = evaluate_model(
        model,
        loader,
        device,
        dataset.target_std,
        bool(config["data"].get("normalize_targets", True)),
    )
    row = {
        "experiment_name": config["experiment_name"],
        "model_type": config["model"]["type"],
        "split": args.split,
        "loss": loss,
        **metrics,
    }
    output_dir = Path(config["output_dir"]) / config["experiment_name"]
    save_json(output_dir / f"metrics_{args.split}.json", row)
    csv_path = output_dir / "metrics.csv"
    exists = csv_path.exists()
    with csv_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    print(row)


if __name__ == "__main__":
    main()

