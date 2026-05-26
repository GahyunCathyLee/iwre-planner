#!/usr/bin/env python3
"""Prepare ego-centric neighbor noise-modeling artifacts from *_nbr.csv logs.

The generated sigma_v* columns use only ego-observable values at inference time:
observed neighbor state, ego state, time, and past observations. Ground truth is
used only for supervised labels and evaluation targets.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


EPS = 1.0e-6

FEATURE_COLUMNS_V1 = [
    "dx_obs",
    "dy_obs",
    "dvx_obs",
    "dvy_obs",
    "distance_obs",
    "sin_bearing_obs",
    "cos_bearing_obs",
]

FEATURE_COLUMNS_V2 = [
    "dx_obs",
    "dy_obs",
    "dvx_obs",
    "dvy_obs",
    "distance_obs",
    "sin_bearing_obs",
    "cos_bearing_obs",
    "relative_speed_obs",
    "closing_rate",
    "ego_vx",
    "ego_vy",
    "ego_speed",
    "vx_obs",
    "vy_obs",
    "neighbor_speed_obs",
    "step_normalized",
    "time_sec_normalized",
    "sigma_v1",
    "sigma_v2",
    "sigma_v3",
]

FEATURE_COLUMNS_V3 = [
    "dx_obs",
    "dy_obs",
    "dvx_obs",
    "dvy_obs",
    "distance_obs",
    "sin_bearing_obs",
    "cos_bearing_obs",
    "relative_speed_obs",
    "closing_rate",
    "ego_speed",
    "neighbor_speed_obs",
    "delta_time",
]

TARGET_COLUMNS_REL = ["noise_dx", "noise_dy", "noise_vx", "noise_vy"]
OBS_STATE_COLUMNS_REL = ["dx_obs", "dy_obs", "vx_obs", "vy_obs"]
TRUE_STATE_COLUMNS_REL = ["dx_true", "dy_true", "vx_gt", "vy_gt"]

DERIVED_COLUMNS = [
    "dx_true",
    "dy_true",
    "noise_dx",
    "noise_dy",
    "noise_vx",
    "noise_vy",
    "noise_r",
    "noise_theta",
    "dvx_obs",
    "dvy_obs",
    "distance_obs",
    "bearing_obs",
    "sin_bearing_obs",
    "cos_bearing_obs",
    "ego_speed",
    "neighbor_speed_obs",
    "relative_speed_obs",
    "closing_rate",
    "delta_time",
    "step_normalized",
    "time_sec_normalized",
    "cv_position_residual",
    "cv_velocity_residual",
    "sigma_v1",
    "sigma_v2",
    "sigma_v3",
    "risk_v1",
    "risk_v2",
    "risk_v3",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create augmented nbr.csv files and ML-ready metadata for noise modeling."
    )
    parser.add_argument("--input-dir", default="outputs/logs")
    parser.add_argument("--output-dir", default="outputs/noise_modeling")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--pattern", default="*_nbr.csv")
    parser.add_argument("--in-place", action="store_true", help="Overwrite source nbr.csv files.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N files. Useful for smoke tests.",
    )
    parser.add_argument("--range-ref", type=float, default=60.0)
    parser.add_argument("--cv-position-ref", type=float, default=1.0)
    parser.add_argument("--cv-velocity-ref", type=float, default=1.5)
    parser.add_argument("--ema-alpha", type=float, default=0.30)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=20260526)
    return parser.parse_args()


def to_float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "" or value is None:
        return default
    return float(value)


def fmt(value: float) -> str:
    if not math.isfinite(value):
        value = 0.0
    return f"{value:.6f}"


def clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def smooth_sigma_from_residual(position_error: float, velocity_error: float, p_ref: float, v_ref: float) -> float:
    score = (position_error / max(p_ref, EPS)) ** 2 + (velocity_error / max(v_ref, EPS)) ** 2
    return clip01(1.0 - math.exp(-0.5 * score))


def run_id_from_path(path: Path) -> str:
    name = path.name
    return name[:-8] if name.endswith("_nbr.csv") else path.stem


def split_episode_ids(
    episode_ids: Sequence[str],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, str]:
    rng = np.random.default_rng(seed)
    ids = np.array(sorted(set(episode_ids)), dtype=object)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    split = {}
    for episode_id in ids[:n_train]:
        split[str(episode_id)] = "train"
    for episode_id in ids[n_train : n_train + n_val]:
        split[str(episode_id)] = "val"
    for episode_id in ids[n_train + n_val :]:
        split[str(episode_id)] = "test"
    return split


def add_static_derived_fields(rows: List[Dict[str, str]], args_dict: Dict[str, float]) -> None:
    max_step = max((to_float(row, "step") for row in rows), default=0.0)
    max_time = max((to_float(row, "time_sec") for row in rows), default=0.0)
    range_ref = float(args_dict["range_ref"])

    for row in rows:
        ego_x = to_float(row, "ego_x")
        ego_y = to_float(row, "ego_y")
        ego_vx = to_float(row, "ego_vx")
        ego_vy = to_float(row, "ego_vy")
        x_gt = to_float(row, "x_gt")
        y_gt = to_float(row, "y_gt")
        vx_gt = to_float(row, "vx_gt")
        vy_gt = to_float(row, "vy_gt")
        dx_obs = to_float(row, "dx_obs")
        dy_obs = to_float(row, "dy_obs")
        vx_obs = to_float(row, "vx_obs")
        vy_obs = to_float(row, "vy_obs")

        dx_true = x_gt - ego_x
        dy_true = y_gt - ego_y
        noise_dx = dx_obs - dx_true
        noise_dy = dy_obs - dy_true
        noise_vx = vx_obs - vx_gt
        noise_vy = vy_obs - vy_gt

        distance_obs = math.hypot(dx_obs, dy_obs)
        bearing_obs = math.atan2(dy_obs, dx_obs)
        distance_true = math.hypot(dx_true, dy_true)
        bearing_true = math.atan2(dy_true, dx_true)
        dvx_obs = vx_obs - ego_vx
        dvy_obs = vy_obs - ego_vy
        ego_speed = math.hypot(ego_vx, ego_vy)
        neighbor_speed_obs = math.hypot(vx_obs, vy_obs)
        relative_speed_obs = math.hypot(dvx_obs, dvy_obs)
        closing_rate = (dx_obs * dvx_obs + dy_obs * dvy_obs) / max(distance_obs, EPS)
        sigma_v1 = clip01(1.0 - math.exp(-0.5 * (distance_obs / max(range_ref, EPS)) ** 2))
        alpha = to_float(row, "alpha")

        row["dx_true"] = fmt(dx_true)
        row["dy_true"] = fmt(dy_true)
        row["noise_dx"] = fmt(noise_dx)
        row["noise_dy"] = fmt(noise_dy)
        row["noise_vx"] = fmt(noise_vx)
        row["noise_vy"] = fmt(noise_vy)
        row["noise_r"] = fmt(distance_obs - distance_true)
        row["noise_theta"] = fmt(wrap_angle(bearing_obs - bearing_true))
        row["dvx_obs"] = fmt(dvx_obs)
        row["dvy_obs"] = fmt(dvy_obs)
        row["distance_obs"] = fmt(distance_obs)
        row["bearing_obs"] = fmt(bearing_obs)
        row["sin_bearing_obs"] = fmt(math.sin(bearing_obs))
        row["cos_bearing_obs"] = fmt(math.cos(bearing_obs))
        row["ego_speed"] = fmt(ego_speed)
        row["neighbor_speed_obs"] = fmt(neighbor_speed_obs)
        row["relative_speed_obs"] = fmt(relative_speed_obs)
        row["closing_rate"] = fmt(closing_rate)
        row["step_normalized"] = fmt(to_float(row, "step") / max(max_step, 1.0))
        row["time_sec_normalized"] = fmt(to_float(row, "time_sec") / max(max_time, EPS))
        row["sigma_v1"] = fmt(sigma_v1)
        row["risk_v1"] = fmt(sigma_v1 * alpha)


def add_history_sigma_fields(rows: List[Dict[str, str]], args_dict: Dict[str, float]) -> None:
    p_ref = float(args_dict["cv_position_ref"])
    v_ref = float(args_dict["cv_velocity_ref"])
    ema_alpha = float(args_dict["ema_alpha"])
    grouped: Dict[Tuple[str, str], List[Dict[str, str]]] = {}

    for row in rows:
        key = (row.get("run_id", ""), row.get("neighbor_id", row.get("vehicle_id", "")))
        grouped.setdefault(key, []).append(row)

    for trajectory in grouped.values():
        trajectory.sort(key=lambda item: (to_float(item, "step"), to_float(item, "time_sec")))
        prev: Optional[Dict[str, str]] = None
        ema = 0.0
        for row in trajectory:
            if prev is None:
                delta_time = 0.0
                pos_residual = 0.0
                vel_residual = 0.0
                sigma_v2 = 0.0
                ema = 0.0
            else:
                delta_time = max(0.0, to_float(row, "time_sec") - to_float(prev, "time_sec"))
                pred_dx = to_float(prev, "dx_obs") + to_float(prev, "dvx_obs") * delta_time
                pred_dy = to_float(prev, "dy_obs") + to_float(prev, "dvy_obs") * delta_time
                pos_residual = math.hypot(to_float(row, "dx_obs") - pred_dx, to_float(row, "dy_obs") - pred_dy)
                vel_residual = math.hypot(
                    to_float(row, "dvx_obs") - to_float(prev, "dvx_obs"),
                    to_float(row, "dvy_obs") - to_float(prev, "dvy_obs"),
                )
                sigma_v2 = smooth_sigma_from_residual(pos_residual, vel_residual, p_ref, v_ref)
                ema = (1.0 - ema_alpha) * ema + ema_alpha * sigma_v2

            alpha = to_float(row, "alpha")
            row["delta_time"] = fmt(delta_time)
            row["cv_position_residual"] = fmt(pos_residual)
            row["cv_velocity_residual"] = fmt(vel_residual)
            row["sigma_v2"] = fmt(sigma_v2)
            row["sigma_v3"] = fmt(ema)
            row["risk_v2"] = fmt(sigma_v2 * alpha)
            row["risk_v3"] = fmt(ema * alpha)
            prev = row


def write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def process_one_file(task: Tuple[str, str, bool, Dict[str, float]]) -> Dict[str, object]:
    source_s, output_root_s, in_place, args_dict = task
    source = Path(source_s)
    output_root = Path(output_root_s)
    with source.open(newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        original_fields = list(reader.fieldnames or [])

    add_static_derived_fields(rows, args_dict)
    add_history_sigma_fields(rows, args_dict)

    fieldnames = original_fields + [name for name in DERIVED_COLUMNS if name not in original_fields]
    output_path = source if in_place else output_root / "augmented_logs" / source.name
    write_csv(output_path, rows, fieldnames)

    return {
        "source": str(source),
        "output": str(output_path),
        "run_id": rows[0].get("run_id", run_id_from_path(source)) if rows else run_id_from_path(source),
        "rows": len(rows),
        "vehicles": len({row.get("neighbor_id", row.get("vehicle_id", "")) for row in rows}),
    }


def load_numeric_matrix(paths: Sequence[Path], columns: Sequence[str]) -> np.ndarray:
    values: List[List[float]] = []
    for path in paths:
        with path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                values.append([to_float(row, column) for column in columns])
    if not values:
        return np.empty((0, len(columns)), dtype=np.float64)
    return np.asarray(values, dtype=np.float64)


def write_normalization_stats(output_root: Path, train_paths: Sequence[Path]) -> None:
    stats = {
        "feature_columns_v1": FEATURE_COLUMNS_V1,
        "feature_columns_v2": FEATURE_COLUMNS_V2,
        "feature_columns_v3": FEATURE_COLUMNS_V3,
        "target_columns_rel": TARGET_COLUMNS_REL,
        "obs_state_columns_rel": OBS_STATE_COLUMNS_REL,
        "true_state_columns_rel": TRUE_STATE_COLUMNS_REL,
    }
    for name, columns in [
        ("features_v1", FEATURE_COLUMNS_V1),
        ("features_v2", FEATURE_COLUMNS_V2),
        ("features_v3", FEATURE_COLUMNS_V3),
        ("target_rel", TARGET_COLUMNS_REL),
    ]:
        matrix = load_numeric_matrix(train_paths, columns)
        if matrix.size == 0:
            mean = [0.0 for _ in columns]
            std = [1.0 for _ in columns]
        else:
            mean = matrix.mean(axis=0)
            std = matrix.std(axis=0)
            std = np.where(std < 1.0e-8, 1.0, std)
        stats[name] = {
            "columns": list(columns),
            "mean": [float(x) for x in mean],
            "std": [float(x) for x in std],
        }
    with (output_root / "normalization_stats.json").open("w") as fh:
        json.dump(stats, fh, indent=2, sort_keys=True)


def write_manifest(output_root: Path, results: Sequence[Dict[str, object]], args: argparse.Namespace) -> None:
    episode_ids = [str(result["run_id"]) for result in results]
    split_by_episode = split_episode_ids(
        episode_ids,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.split_seed,
    )

    manifest_path = output_root / "manifest.csv"
    with manifest_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["run_id", "split", "rows", "vehicles", "source_path", "augmented_path"],
        )
        writer.writeheader()
        for result in sorted(results, key=lambda item: str(item["run_id"])):
            run_id = str(result["run_id"])
            writer.writerow({
                "run_id": run_id,
                "split": split_by_episode[run_id],
                "rows": result["rows"],
                "vehicles": result["vehicles"],
                "source_path": result["source"],
                "augmented_path": result["output"],
            })

    split_counts = {"train": 0, "val": 0, "test": 0}
    split_rows = {"train": 0, "val": 0, "test": 0}
    for result in results:
        split = split_by_episode[str(result["run_id"])]
        split_counts[split] += 1
        split_rows[split] += int(result["rows"])

    metadata = {
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "file_count": len(results),
        "row_count": sum(int(result["rows"]) for result in results),
        "split_counts": split_counts,
        "split_rows": split_rows,
        "sigma_definitions": {
            "sigma_v1": "range-only heuristic: 1 - exp(-0.5 * (distance_obs / range_ref)^2)",
            "sigma_v2": "constant-velocity innovation from current and previous ego-observable states",
            "sigma_v3": "EMA-smoothed sigma_v2",
        },
        "parameters": {
            "range_ref": args.range_ref,
            "cv_position_ref": args.cv_position_ref,
            "cv_velocity_ref": args.cv_velocity_ref,
            "ema_alpha": args.ema_alpha,
            "split_seed": args.split_seed,
        },
    }
    with (output_root / "metadata.json").open("w") as fh:
        json.dump(metadata, fh, indent=2, sort_keys=True)

    train_paths = [
        Path(result["output"])
        for result in results
        if split_by_episode[str(result["run_id"])] == "train"
    ]
    write_normalization_stats(output_root, train_paths)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    sources = sorted(input_dir.glob(args.pattern))
    if args.limit is not None:
        sources = sources[: args.limit]
    if not sources:
        raise SystemExit(f"No files matched {input_dir / args.pattern}")

    args_dict = {
        "range_ref": args.range_ref,
        "cv_position_ref": args.cv_position_ref,
        "cv_velocity_ref": args.cv_velocity_ref,
        "ema_alpha": args.ema_alpha,
    }
    tasks = [(str(path), str(output_root), bool(args.in_place), args_dict) for path in sources]
    results: List[Dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(process_one_file, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if index == 1 or index % 25 == 0 or index == len(futures):
                print(f"[INFO] processed {index}/{len(futures)} files")

    write_manifest(output_root, results, args)
    print(f"[INFO] wrote augmented logs and metadata under {output_root}")


if __name__ == "__main__":
    main()
