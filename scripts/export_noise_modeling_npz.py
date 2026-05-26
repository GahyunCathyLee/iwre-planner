#!/usr/bin/env python3
"""Export augmented noise-modeling CSV files into run-level NumPy shards."""

from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


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
SIGMA_COLUMNS = ["sigma_v1", "sigma_v2", "sigma_v3"]
RISK_COLUMNS = ["risk_v1", "risk_v2", "risk_v3"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export noise-modeling augmented CSVs to NPZ shards.")
    parser.add_argument("--manifest", default="outputs/noise_modeling/manifest.csv")
    parser.add_argument("--output-dir", default="outputs/noise_modeling/npz")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--history-length", type=int, default=10)
    parser.add_argument("--no-sequences", action="store_true")
    return parser.parse_args()


def to_float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "" or value is None:
        return default
    return float(value)


def numeric_matrix(rows: Sequence[Dict[str, str]], columns: Sequence[str]) -> np.ndarray:
    return np.asarray([[to_float(row, column) for column in columns] for row in rows], dtype=np.float32)


def string_array(rows: Sequence[Dict[str, str]], column: str) -> np.ndarray:
    return np.asarray([row.get(column, "") for row in rows])


def build_sequences(
    rows: Sequence[Dict[str, str]],
    base_features: np.ndarray,
    history_length: int,
) -> np.ndarray:
    if history_length <= 1:
        return base_features[:, None, :]

    n_rows, feature_dim = base_features.shape
    sequences = np.empty((n_rows, history_length, feature_dim), dtype=np.float32)
    grouped: Dict[Tuple[str, str], List[int]] = {}
    for index, row in enumerate(rows):
        key = (row.get("run_id", ""), row.get("neighbor_id", row.get("vehicle_id", "")))
        grouped.setdefault(key, []).append(index)

    for indices in grouped.values():
        indices.sort(key=lambda idx: (to_float(rows[idx], "step"), to_float(rows[idx], "time_sec")))
        for local_pos, row_index in enumerate(indices):
            start = max(0, local_pos - history_length + 1)
            history_indices = indices[start : local_pos + 1]
            if len(history_indices) < history_length:
                pad_count = history_length - len(history_indices)
                history_indices = [history_indices[0]] * pad_count + history_indices
            sequences[row_index] = base_features[history_indices]
    return sequences


def export_one(task: Tuple[Dict[str, str], str, int, bool]) -> Dict[str, object]:
    manifest_row, output_dir_s, history_length, no_sequences = task
    source = Path(manifest_row["augmented_path"])
    split = manifest_row["split"]
    run_id = manifest_row["run_id"]
    output_dir = Path(output_dir_s) / split
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{run_id}.npz"

    with source.open(newline="") as fh:
        rows = list(csv.DictReader(fh))

    features_v1 = numeric_matrix(rows, FEATURE_COLUMNS_V1)
    features_v2 = numeric_matrix(rows, FEATURE_COLUMNS_V2)
    features_v3 = numeric_matrix(rows, FEATURE_COLUMNS_V3)
    arrays = {
        "features_v1": features_v1,
        "features_v2": features_v2,
        "features_v3": features_v3,
        "target_rel": numeric_matrix(rows, TARGET_COLUMNS_REL),
        "obs_state_rel": numeric_matrix(rows, OBS_STATE_COLUMNS_REL),
        "true_state_rel": numeric_matrix(rows, TRUE_STATE_COLUMNS_REL),
        "sigma": numeric_matrix(rows, SIGMA_COLUMNS),
        "risk": numeric_matrix(rows, RISK_COLUMNS),
        "step": numeric_matrix(rows, ["step"]).reshape(-1).astype(np.int32),
        "time_sec": numeric_matrix(rows, ["time_sec"]).reshape(-1),
        "neighbor_id": string_array(rows, "neighbor_id"),
        "actor_id": string_array(rows, "actor_id"),
    }
    if not no_sequences:
        arrays["sequence_features_v3"] = build_sequences(rows, features_v3, history_length)

    np.savez_compressed(output_path, **arrays)
    return {"run_id": run_id, "split": split, "rows": len(rows), "path": str(output_path)}


def read_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(manifest_path)
    tasks = [(row, str(output_dir), args.history_length, bool(args.no_sequences)) for row in rows]

    results: List[Dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(export_one, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if index == 1 or index % 25 == 0 or index == len(futures):
                print(f"[INFO] exported {index}/{len(futures)} shards")

    summary = {
        "manifest": str(manifest_path),
        "output_dir": str(output_dir),
        "history_length": args.history_length,
        "with_sequences": not args.no_sequences,
        "shard_count": len(results),
        "row_count": sum(int(result["rows"]) for result in results),
        "columns": {
            "features_v1": FEATURE_COLUMNS_V1,
            "features_v2": FEATURE_COLUMNS_V2,
            "features_v3": FEATURE_COLUMNS_V3,
            "target_rel": TARGET_COLUMNS_REL,
            "obs_state_rel": OBS_STATE_COLUMNS_REL,
            "true_state_rel": TRUE_STATE_COLUMNS_REL,
            "sigma": SIGMA_COLUMNS,
            "risk": RISK_COLUMNS,
        },
    }
    with (output_dir / "npz_manifest.json").open("w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    print(f"[INFO] wrote NPZ shards under {output_dir}")


if __name__ == "__main__":
    main()
