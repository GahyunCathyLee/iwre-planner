#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


SLOT_NAMES = [
    "preceding",
    "following",
    "leftPreceding",
    "leftAlongside",
    "leftFollowing",
    "rightPreceding",
    "rightAlongside",
    "rightFollowing",
]
SLOT_TO_INDEX = {name: index for index, name in enumerate(SLOT_NAMES)}
FEATURE_NAMES = [
    "dx_obs",
    "dy_obs",
    "dvx_obs",
    "dvy_obs",
    "distance_obs",
    "closing_rate",
    "slot_index_norm",
    "valid",
    "is_ego",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GRIP-style interaction alpha dataset from IWRE logs.")
    parser.add_argument("--logs-dir", default="outputs/logs")
    parser.add_argument("--out-dir", default="outputs/interaction_grip")
    parser.add_argument("--history", type=int, default=6)
    parser.add_argument("--future", type=int, default=15)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def to_float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        return default if value == "" else float(value)
    except (TypeError, ValueError):
        return default


def to_int(value: str, default: int = 0) -> int:
    try:
        return default if value == "" else int(float(value))
    except (TypeError, ValueError):
        return default


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def build_raw_adj(valid_mask: np.ndarray) -> np.ndarray:
    adj = np.eye(9, dtype=np.float32)
    valid_nodes = np.flatnonzero(valid_mask > 0.0)
    for node in valid_nodes:
        adj[0, node] = 1.0
        adj[node, 0] = 1.0
    return adj


def make_sample(
    ego_by_step: Dict[int, Dict[str, str]],
    nbr_by_step_vehicle: Dict[Tuple[int, int], Dict[str, str]],
    t0: int,
    history: int,
    future: int,
) -> Dict[str, np.ndarray] | None:
    hist_steps = list(range(t0 - history + 1, t0 + 1))
    fut_steps = list(range(t0 + 1, t0 + future + 1))
    if any(step not in ego_by_step for step in hist_steps + fut_steps):
        return None

    x = np.zeros((len(FEATURE_NAMES), history, 9), dtype=np.float32)
    valid_mask = np.zeros(9, dtype=np.float32)
    alpha_heuristic = np.zeros(8, dtype=np.float32)
    last_ego = ego_by_step[t0]
    last_ego_x = to_float(last_ego, "ego_x")
    last_ego_y = to_float(last_ego, "ego_y")

    for hi, step in enumerate(hist_steps):
        ego = ego_by_step[step]
        ego_vx = to_float(ego, "ego_vx")
        ego_vy = to_float(ego, "ego_vy")
        x[2, hi, 0] = ego_vx
        x[3, hi, 0] = ego_vy
        x[7, hi, 0] = 1.0
        x[8, hi, 0] = 1.0
        valid_mask[0] = 1.0

        for slot_name, slot_index in SLOT_TO_INDEX.items():
            vehicle_id = to_int(ego.get(slot_name, ""))
            if vehicle_id <= 0:
                continue
            nbr = nbr_by_step_vehicle.get((step, vehicle_id))
            if nbr is None:
                continue
            node = slot_index + 1
            dx = to_float(nbr, "dx_obs")
            dy = to_float(nbr, "dy_obs")
            dvx = to_float(nbr, "vx_obs") - ego_vx
            dvy = to_float(nbr, "vy_obs") - ego_vy
            distance = math.hypot(dx, dy)
            closing = (dx * dvx + dy * dvy) / max(distance, 1.0e-6)
            x[:, hi, node] = [
                dx,
                dy,
                dvx,
                dvy,
                distance,
                closing,
                slot_index / max(len(SLOT_NAMES) - 1, 1),
                1.0,
                0.0,
            ]
            if step == t0:
                valid_mask[node] = 1.0
                alpha_heuristic[slot_index] = to_float(nbr, "alpha")

    target = np.zeros((future, 2), dtype=np.float32)
    for fi, step in enumerate(fut_steps):
        ego = ego_by_step[step]
        target[fi, 0] = to_float(ego, "ego_x") - last_ego_x
        target[fi, 1] = to_float(ego, "ego_y") - last_ego_y

    return {
        "x": x,
        "adj": build_raw_adj(valid_mask),
        "target": target,
        "valid_mask": valid_mask,
        "alpha_heuristic": alpha_heuristic,
    }


def write_shards(out_dir: Path, split: str, samples: List[Dict[str, np.ndarray]], shard_size: int) -> int:
    split_dir = out_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for shard_index, start in enumerate(range(0, len(samples), shard_size)):
        chunk = samples[start : start + shard_size]
        if not chunk:
            continue
        np.savez_compressed(
            split_dir / f"shard_{shard_index:05d}.npz",
            x=np.stack([s["x"] for s in chunk]),
            adj=np.stack([s["adj"] for s in chunk]),
            target=np.stack([s["target"] for s in chunk]),
            valid_mask=np.stack([s["valid_mask"] for s in chunk]),
            alpha_heuristic=np.stack([s["alpha_heuristic"] for s in chunk]),
        )
        count += len(chunk)
    return count


def main() -> None:
    args = parse_args()
    logs_dir = Path(args.logs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_ids = sorted(path.name[: -len("_ego.csv")] for path in logs_dir.glob("*_ego.csv"))
    if args.max_runs is not None:
        run_ids = run_ids[: args.max_runs]

    samples_by_run: Dict[str, List[Dict[str, np.ndarray]]] = defaultdict(list)
    for run_id in run_ids:
        ego_path = logs_dir / f"{run_id}_ego.csv"
        nbr_path = logs_dir / f"{run_id}_nbr.csv"
        if not nbr_path.exists():
            continue
        ego_rows = read_csv_rows(ego_path)
        nbr_rows = read_csv_rows(nbr_path)
        ego_by_step = {to_int(row["step"]): row for row in ego_rows}
        nbr_by_step_vehicle = {
            (to_int(row["step"]), to_int(row.get("vehicle_id", row.get("neighbor_id", "")))): row
            for row in nbr_rows
        }
        steps = sorted(ego_by_step)
        if not steps:
            continue
        start = min(steps) + args.history - 1
        end = max(steps) - args.future
        for t0 in range(start, end + 1, args.stride):
            sample = make_sample(ego_by_step, nbr_by_step_vehicle, t0, args.history, args.future)
            if sample is not None:
                samples_by_run[run_id].append(sample)

    rng = random.Random(args.seed)
    split_runs = list(samples_by_run)
    rng.shuffle(split_runs)
    n = len(split_runs)
    split_map = {
        "train": split_runs[: int(n * 0.7)],
        "val": split_runs[int(n * 0.7) : int(n * 0.8)],
        "test": split_runs[int(n * 0.8) :],
    }

    counts = {}
    for split, split_run_ids in split_map.items():
        split_samples = [sample for run_id in split_run_ids for sample in samples_by_run[run_id]]
        counts[split] = write_shards(out_dir, split, split_samples, args.shard_size)

    metadata = {
        "history": args.history,
        "future": args.future,
        "stride": args.stride,
        "feature_names": FEATURE_NAMES,
        "slot_names": SLOT_NAMES,
        "num_runs": len(samples_by_run),
        "sample_counts": counts,
    }
    with (out_dir / "metadata.json").open("w") as fh:
        json.dump(metadata, fh, indent=2)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

