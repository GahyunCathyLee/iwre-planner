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
    parser.add_argument("--output-dir", default=None, help="Override checkpoint config output_dir for metrics files.")
    return parser.parse_args()


def format_metric(value) -> str:
    if isinstance(value, float):
        return f"{value: .6f}"
    return str(value)


def print_metrics(row: dict, checkpoint_path: str, output_dir: Path) -> None:
    print()
    print("=" * 72)
    print("Noise Modeling Evaluation")
    print("=" * 72)
    print(f"checkpoint : {checkpoint_path}")
    print(f"experiment : {row['experiment_name']}")
    print(f"model      : {row['model_type']}")
    print(f"split      : {row['split']}")
    print(f"metrics dir: {output_dir}")
    print("-" * 72)

    groups = [
        (
            "Core",
            [
                "loss",
                "nll",
                "predicted_sigma_mean",
            ],
        ),
        (
            "Noise Prediction RMSE",
            [
                "noise_rmse",
                "noise_rmse_dx",
                "noise_rmse_dy",
                "noise_rmse_vx",
                "noise_rmse_vy",
            ],
        ),
        (
            "Correction",
            [
                "observed_rmse",
                "corrected_rmse",
                "improvement_ratio",
                "worsening_rate",
            ],
        ),
    ]
    for title, keys in groups:
        print(title)
        for key in keys:
            if key in row:
                print(f"  {key:<24} {format_metric(row[key])}")
        print()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    config = checkpoint["config"]
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
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
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / f"metrics_{args.split}.json", row)
    csv_path = output_dir / "metrics.csv"
    exists = csv_path.exists()
    with csv_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    print_metrics(row, args.checkpoint, output_dir)


if __name__ == "__main__":
    main()
