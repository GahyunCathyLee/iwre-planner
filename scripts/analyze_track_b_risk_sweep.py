#!/usr/bin/env python3
"""Analyze a track_b risk-gain/min-distance-scale sweep and pick a follow-up config."""

from __future__ import annotations

import argparse
import csv
import math
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


METRICS = [
    "mean_speed",
    "path_distance",
    "mean_brake",
    "brake_step_frac",
    "mean_planner_risk",
    "collision_run",
    "near_miss_true_run",
]


def to_float(value: str, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def to_int(value: str, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def summarize_ego(path: Path) -> dict[str, float]:
    rows = 0
    speed = throttle = brake = planner_risk = 0.0
    brake_steps = 0
    prev_xy: tuple[float, float] | None = None
    path_distance = 0.0

    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            rows += 1
            ego_speed = to_float(row.get("ego_speed", ""))
            row_brake = to_float(row.get("brake", ""))
            x = to_float(row.get("ego_x", ""))
            y = to_float(row.get("ego_y", ""))
            speed += ego_speed
            throttle += to_float(row.get("throttle", ""))
            brake += row_brake
            planner_risk += to_float(row.get("planner_risk", ""))
            if row_brake > 0.05:
                brake_steps += 1
            if prev_xy is not None:
                path_distance += math.hypot(x - prev_xy[0], y - prev_xy[1])
            prev_xy = (x, y)

    denom = max(rows, 1)
    return {
        "ego_steps": rows,
        "mean_speed": speed / denom,
        "path_distance": path_distance,
        "mean_throttle": throttle / denom,
        "mean_brake": brake / denom,
        "brake_step_frac": brake_steps / denom,
        "mean_planner_risk": planner_risk / denom,
    }


def summarize_nbr(path: Path) -> dict[str, float]:
    collision = 0
    near_miss = 0
    rows = 0
    risk = 0.0
    sigma = 0.0
    alpha = 0.0

    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            rows += 1
            collision = max(collision, to_int(row.get("collision", "")))
            near_miss = max(near_miss, to_int(row.get("near_miss_true", "")))
            risk += to_float(row.get("risk", ""))
            sigma += to_float(row.get("planner_sigma", ""))
            alpha += to_float(row.get("alpha", ""))

    denom = max(rows, 1)
    return {
        "nbr_rows": rows,
        "collision_run": collision,
        "near_miss_true_run": near_miss,
        "mean_risk_nbr": risk / denom,
        "mean_sigma_nbr": sigma / denom,
        "mean_alpha_nbr": alpha / denom,
    }


def expected_keys(args: argparse.Namespace) -> set[tuple[str, str, str, str, str]]:
    scenarios = args.scenario_ids.split()
    risk_gains = args.risk_gains.split()
    min_distance_scales = args.min_distance_scales.split()
    seeds = args.seeds.split()
    planners = args.planners.split()
    return {
        (scenario_id, planner, seed, risk_gain, min_distance_scale)
        for scenario_id in scenarios
        for planner in planners
        for seed in seeds
        for risk_gain in risk_gains
        for min_distance_scale in min_distance_scales
    }


def present_keys(rows: list[dict[str, str]]) -> set[tuple[str, str, str, str, str]]:
    keys = set()
    for row in rows:
        if row.get("sigma_source") != "track_b":
            continue
        if row.get("status") != "ok":
            continue
        keys.add(
            (
                str(to_int(row.get("scenario_id", ""))),
                row.get("planner", ""),
                str(to_int(row.get("seed", ""))),
                str(to_float(row.get("risk_gain", ""))),
                str(to_float(row.get("min_distance_scale", ""))),
            )
        )
    return keys


def wait_for_manifest(args: argparse.Namespace) -> None:
    expected = expected_keys(args)
    deadline = time.time() + args.wait_timeout_sec
    while True:
        rows = read_manifest(Path(args.manifest))
        present = present_keys(rows)
        missing = expected - present
        print(f"[INFO] sweep progress: {len(expected) - len(missing)}/{len(expected)} ok runs")
        if not missing:
            return
        if time.time() >= deadline:
            raise TimeoutError(f"Timed out waiting for sweep completion; missing {len(missing)} runs")
        time.sleep(args.poll_sec)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def analyze(args: argparse.Namespace) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    manifest_rows = read_manifest(Path(args.manifest))
    latest = {}
    for row in manifest_rows:
        latest[row["run_id"]] = row

    run_rows = []
    for row in latest.values():
        if row.get("sigma_source") != "track_b" or row.get("status") != "ok":
            continue
        if row.get("planner") not in {"B2", "B3"}:
            continue
        ego_path = Path(row["ego_log"])
        nbr_path = Path(row["nbr_log"])
        if not ego_path.exists() or not nbr_path.exists():
            continue
        item: dict[str, object] = {
            "run_id": row["run_id"],
            "scenario_id": to_int(row["scenario_id"]),
            "planner": row["planner"],
            "seed": to_int(row["seed"]),
            "risk_gain": to_float(row["risk_gain"]),
            "min_distance_scale": to_float(row["min_distance_scale"]),
        }
        item.update(summarize_ego(ego_path))
        item.update(summarize_nbr(nbr_path))
        run_rows.append(item)

    by_pair = {
        (
            row["scenario_id"],
            row["seed"],
            row["risk_gain"],
            row["min_distance_scale"],
            row["planner"],
        ): row
        for row in run_rows
    }

    pair_rows = []
    combo_pairs: dict[tuple[float, float], list[dict[str, object]]] = defaultdict(list)
    for row in run_rows:
        if row["planner"] != "B3":
            continue
        key = (row["scenario_id"], row["seed"], row["risk_gain"], row["min_distance_scale"], "B2")
        peer = by_pair.get(key)
        if peer is None:
            continue
        pair = {
            "scenario_id": row["scenario_id"],
            "seed": row["seed"],
            "risk_gain": row["risk_gain"],
            "min_distance_scale": row["min_distance_scale"],
        }
        for metric in METRICS:
            pair[f"{metric}_diff"] = float(row[metric]) - float(peer[metric])
        pair_rows.append(pair)
        combo_pairs[(float(row["risk_gain"]), float(row["min_distance_scale"]))].append(pair)

    summary_rows = []
    for (risk_gain, min_distance_scale), pairs in sorted(combo_pairs.items()):
        summary: dict[str, object] = {
            "risk_gain": risk_gain,
            "min_distance_scale": min_distance_scale,
            "pairs": len(pairs),
        }
        for metric in METRICS:
            values = [float(row[f"{metric}_diff"]) for row in pairs]
            summary[f"{metric}_diff_mean"] = mean(values)
            summary[f"{metric}_diff_std"] = pstdev(values) if len(values) > 1 else 0.0

        safety_penalty = 40.0 * summary["collision_run_diff_mean"] + 20.0 * summary["near_miss_true_run_diff_mean"]
        efficiency_gain = summary["mean_speed_diff_mean"] + summary["path_distance_diff_mean"] / 30.0
        comfort_gain = -3.0 * summary["mean_brake_diff_mean"] - summary["brake_step_frac_diff_mean"]
        summary["strict_b3_better"] = (
            summary["collision_run_diff_mean"] <= 0.0
            and summary["near_miss_true_run_diff_mean"] <= 0.0
            and summary["mean_speed_diff_mean"] > 0.0
            and summary["path_distance_diff_mean"] > 0.0
            and summary["mean_brake_diff_mean"] < 0.0
        )
        summary["score"] = efficiency_gain + comfort_gain - safety_penalty
        summary_rows.append(summary)

    strict = [row for row in summary_rows if row["strict_b3_better"]]
    candidates = strict if strict else summary_rows
    best = max(candidates, key=lambda row: (float(row["score"]), int(row["pairs"])))
    best = dict(best)
    best["selection_mode"] = "strict" if strict else "score"
    return best, run_rows, summary_rows


def write_env(path: Path, best: dict[str, object]) -> None:
    risk_gain = str(best["risk_gain"]).replace(".", "p")
    min_distance_scale = str(best["min_distance_scale"]).replace(".", "p")
    tag = f"trackb_rg{risk_gain}_mds{min_distance_scale}"
    lines = [
        f"BEST_RISK_GAIN={best['risk_gain']}",
        f"BEST_MIN_DISTANCE_SCALE={best['min_distance_scale']}",
        f"BEST_SWEEP_TAG={tag}",
        f"BEST_SELECTION_MODE={best['selection_mode']}",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="outputs/logs/track_b_risk_sweep_manifest.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/track_b_risk_sweep")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--wait-timeout-sec", type=int, default=86400)
    parser.add_argument("--poll-sec", type=int, default=120)
    parser.add_argument("--scenario-ids", default="1 4 5 8 9 10")
    parser.add_argument("--planners", default="B2 B3")
    parser.add_argument("--risk-gains", default="0.5 0.7 0.9")
    parser.add_argument("--min-distance-scales", default="0.1 0.2")
    parser.add_argument("--seeds", default="1 2 3 4 5 6 7 8 9 10")
    args = parser.parse_args()

    if args.wait:
        wait_for_manifest(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best, run_rows, summary_rows = analyze(args)

    write_csv(output_dir / "run_summary.csv", run_rows, list(run_rows[0].keys()))
    write_csv(output_dir / "combo_summary.csv", summary_rows, list(summary_rows[0].keys()))
    write_env(output_dir / "best.env", best)

    print("[INFO] best track_b sweep config")
    for key in ["risk_gain", "min_distance_scale", "pairs", "selection_mode", "score"]:
        print(f"{key}={best[key]}")
    print(f"[INFO] wrote {output_dir / 'combo_summary.csv'}")
    print(f"[INFO] wrote {output_dir / 'best.env'}")


if __name__ == "__main__":
    main()
