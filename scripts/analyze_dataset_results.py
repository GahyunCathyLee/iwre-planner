#!/usr/bin/env python3
"""Aggregate IWRE dataset-generation logs into run and experiment summaries."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


SIGMA_SOURCES = [
    "estimated",
    "v1",
    "v2",
    "v3",
    "ai_v1",
    "ai_v2",
    "ai_v3",
    "track_a",
    "track_b",
]
PLANNERS = ["B1", "B2", "B3"]


def f(value: str, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def i(value: str, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    latest: dict[str, dict[str, str]] = {}
    for row in rows:
        latest[row["run_id"]] = row
    return list(latest.values())


def summarize_ego(path: Path) -> dict[str, float]:
    rows = 0
    sum_speed = sum_throttle = sum_brake = sum_risk = 0.0
    max_brake = max_risk = 0.0
    brake_steps = 0
    min_front = math.inf
    min_eff_front = math.inf
    prev_xy: tuple[float, float] | None = None
    path_distance = 0.0

    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            rows += 1
            speed = f(row.get("ego_speed", ""))
            throttle = f(row.get("throttle", ""))
            brake = f(row.get("brake", ""))
            risk = f(row.get("planner_risk", ""))
            x = f(row.get("ego_x", ""))
            y = f(row.get("ego_y", ""))
            front = f(row.get("front_distance", ""), math.inf)
            eff_front = f(row.get("effective_front_distance", ""), math.inf)

            sum_speed += speed
            sum_throttle += throttle
            sum_brake += brake
            sum_risk += risk
            max_brake = max(max_brake, brake)
            max_risk = max(max_risk, risk)
            if brake > 0.05:
                brake_steps += 1
            if front > 0:
                min_front = min(min_front, front)
            if eff_front > 0:
                min_eff_front = min(min_eff_front, eff_front)
            if prev_xy is not None:
                path_distance += math.hypot(x - prev_xy[0], y - prev_xy[1])
            prev_xy = (x, y)

    denom = max(rows, 1)
    return {
        "ego_steps": rows,
        "mean_speed": sum_speed / denom,
        "path_distance": path_distance,
        "mean_throttle": sum_throttle / denom,
        "mean_brake": sum_brake / denom,
        "max_brake": max_brake,
        "brake_step_frac": brake_steps / denom,
        "mean_planner_risk": sum_risk / denom,
        "max_planner_risk": max_risk,
        "min_front_distance": 0.0 if min_front == math.inf else min_front,
        "min_effective_front_distance": 0.0 if min_eff_front == math.inf else min_eff_front,
    }


def summarize_nbr(path: Path) -> dict[str, float]:
    rows = 0
    collisions = 0
    near_true = near_obs = 0
    min_ttc_true = math.inf
    min_ttc_obs = math.inf
    sum_sigma = sum_alpha = sum_risk = 0.0
    sum_noise_scale = 0.0
    max_sigma = max_alpha = max_risk = 0.0
    sum_pos_err = sum_vel_err = 0.0
    slot_rows = 0
    slot_risk = slot_sigma = 0.0

    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            rows += 1
            sigma = f(row.get("planner_sigma", ""))
            alpha = f(row.get("alpha", ""))
            risk = f(row.get("risk", ""))
            pos_err = f(row.get("position_error", ""))
            vel_err = f(row.get("velocity_error", ""))
            noise_scale = f(row.get("noise_scale", ""))

            sum_sigma += sigma
            sum_alpha += alpha
            sum_risk += risk
            sum_noise_scale += noise_scale
            sum_pos_err += pos_err
            sum_vel_err += vel_err
            max_sigma = max(max_sigma, sigma)
            max_alpha = max(max_alpha, alpha)
            max_risk = max(max_risk, risk)
            collisions = max(collisions, i(row.get("collision", "")))
            near_true = max(near_true, i(row.get("near_miss_true", "")))
            near_obs = max(near_obs, i(row.get("near_miss_obs", "")))

            ttc_true = f(row.get("ttc_true", ""), math.inf)
            ttc_obs = f(row.get("ttc_obs", ""), math.inf)
            if ttc_true > 0:
                min_ttc_true = min(min_ttc_true, ttc_true)
            if ttc_obs > 0:
                min_ttc_obs = min(min_ttc_obs, ttc_obs)

            if row.get("slot", ""):
                slot_rows += 1
                slot_sigma += sigma
                slot_risk += risk

    denom = max(rows, 1)
    slot_denom = max(slot_rows, 1)
    return {
        "nbr_rows": rows,
        "collision_run": collisions,
        "near_miss_true_run": near_true,
        "near_miss_obs_run": near_obs,
        "min_ttc_true": 0.0 if min_ttc_true == math.inf else min_ttc_true,
        "min_ttc_obs": 0.0 if min_ttc_obs == math.inf else min_ttc_obs,
        "mean_planner_sigma_nbr": sum_sigma / denom,
        "max_planner_sigma_nbr": max_sigma,
        "mean_alpha_nbr": sum_alpha / denom,
        "max_alpha_nbr": max_alpha,
        "mean_risk_nbr": sum_risk / denom,
        "max_risk_nbr": max_risk,
        "mean_position_error": sum_pos_err / denom,
        "mean_velocity_error": sum_vel_err / denom,
        "mean_noise_scale_nbr": sum_noise_scale / denom,
        "slot_rows": slot_rows,
        "mean_slot_sigma": slot_sigma / slot_denom,
        "mean_slot_risk": slot_risk / slot_denom,
    }


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def numeric_summary(rows: list[dict[str, object]], keys: tuple[str, ...], metric_fields: list[str]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)

    out: list[dict[str, object]] = []
    for key_values, group in sorted(grouped.items()):
        item = {key: value for key, value in zip(keys, key_values)}
        item["runs"] = len(group)
        for field in metric_fields:
            values = [float(row[field]) for row in group]
            item[f"{field}_mean"] = mean(values)
            item[f"{field}_std"] = pstdev(values) if len(values) > 1 else 0.0
        out.append(item)
    return out


def build_pairwise(rows: list[dict[str, object]], left: str, right: str) -> list[dict[str, object]]:
    by_key: dict[tuple[object, ...], dict[str, object]] = {}
    for row in rows:
        key = (row["scenario_id"], row["sigma_source"], row["seed"], row["planner"])
        by_key[key] = row

    out = []
    metrics = [
        "mean_speed",
        "path_distance",
        "mean_brake",
        "brake_step_frac",
        "collision_run",
        "near_miss_true_run",
        "mean_planner_risk",
        "mean_risk_nbr",
        "min_effective_front_distance",
    ]
    for row in rows:
        if row["planner"] != left:
            continue
        key = (row["scenario_id"], row["sigma_source"], row["seed"], right)
        peer = by_key.get(key)
        if peer is None:
            continue
        item = {
            "scenario_id": row["scenario_id"],
            "sigma_source": row["sigma_source"],
            "seed": row["seed"],
            "left": left,
            "right": right,
        }
        for metric in metrics:
            item[f"{metric}_diff"] = float(row[metric]) - float(peer[metric])
        out.append(item)
    return out


def make_markdown(
    path: Path,
    completeness: list[dict[str, object]],
    group_rows: list[dict[str, object]],
    pair_rows: list[dict[str, object]],
) -> None:
    def fmt(value: object) -> str:
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    top = sorted(
        group_rows,
        key=lambda row: (
            -float(row["collision_run_mean"]),
            -float(row["near_miss_true_run_mean"]),
            -float(row["mean_speed_mean"]),
        ),
    )
    pair_group = numeric_summary(
        pair_rows,
        ("sigma_source",),
        [
            "mean_speed_diff",
            "path_distance_diff",
            "mean_brake_diff",
            "brake_step_frac_diff",
            "near_miss_true_run_diff",
            "collision_run_diff",
        ],
    )

    lines = [
        "# IWRE Result Analysis",
        "",
        "## Completeness",
        "",
        "| planner | sigma_source | expected_runs | found_runs | missing_runs | missing_files |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in completeness:
        lines.append(
            "| {planner} | {sigma_source} | {expected_runs} | {found_runs} | {missing_runs} | {missing_files} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## Highest Event Rates",
            "",
            "| planner | sigma | runs | collision_rate | near_miss_rate | mean_speed | mean_brake | brake_frac | path_distance |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in top[:18]:
        lines.append(
            "| {planner} | {sigma_source} | {runs} | {collision_run_mean:.3f} | {near_miss_true_run_mean:.3f} | {mean_speed_mean:.3f} | {mean_brake_mean:.3f} | {brake_step_frac_mean:.3f} | {path_distance_mean:.3f} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## B3 Minus B2 Paired Differences",
            "",
            "Positive speed/path means B3 moved more than B2 under the same scenario, sigma source, and seed. Negative brake/event values mean B3 was less conservative or safer on that metric.",
            "",
            "| sigma | pairs | speed_diff | path_diff | brake_diff | brake_frac_diff | near_miss_diff | collision_diff |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in pair_group:
        lines.append(
            "| {sigma_source} | {runs} | {mean_speed_diff_mean:.4f} | {path_distance_diff_mean:.4f} | {mean_brake_diff_mean:.4f} | {brake_step_frac_diff_mean:.4f} | {near_miss_true_run_diff_mean:.4f} | {collision_run_diff_mean:.4f} |".format(
                **row
            )
        )

    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", default="outputs/logs")
    parser.add_argument("--manifest", default="outputs/logs/dataset_generation_manifest.csv")
    parser.add_argument("--output-dir", default="outputs/analysis")
    parser.add_argument("--seeds", default="1-20")
    args = parser.parse_args()

    seed_start, seed_end = [int(part) for part in args.seeds.split("-", 1)]
    seeds = set(range(seed_start, seed_end + 1))
    manifest_path = Path(args.manifest)
    log_root = Path(args.log_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = [
        row
        for row in read_manifest(manifest_path)
        if row.get("planner") in PLANNERS
        and row.get("sigma_source") in SIGMA_SOURCES
        and i(row.get("seed", "")) in seeds
        and row.get("status") == "ok"
    ]
    scenarios = sorted({i(row["scenario_id"]) for row in manifest_rows})
    scenario_by_id = {}
    for row in manifest_rows:
        scenario_by_id.setdefault(i(row["scenario_id"]), row.get("scenario", ""))

    manifest_by_key = {
        (i(row["scenario_id"]), row["planner"], row["sigma_source"], i(row["seed"])): row
        for row in manifest_rows
    }
    expected: list[tuple[int, str, str, int]] = []
    for scenario_id in scenarios:
        for seed in sorted(seeds):
            expected.append((scenario_id, "B1", "estimated", seed))
            for planner in ("B2", "B3"):
                for sigma in SIGMA_SOURCES:
                    expected.append((scenario_id, planner, sigma, seed))
    expected_set = set(expected)

    target_rows = []
    for scenario_id, planner, sigma, seed in sorted(expected_set):
        row = manifest_by_key.get((scenario_id, planner, sigma, seed))
        if planner == "B1":
            run_id = f"s{scenario_id}_B1_seed{seed}"
        else:
            run_id = f"s{scenario_id}_{planner}_sig{sigma}_seed{seed}"
        ego_log = log_root / f"{run_id}_ego.csv"
        nbr_log = log_root / f"{run_id}_nbr.csv"
        if row is not None:
            row = dict(row)
            row["ego_log"] = str(ego_log)
            row["nbr_log"] = str(nbr_log)
        else:
            row = {
                "run_id": run_id,
                "planner": planner,
                "sigma_source": sigma,
                "sigma_model_path": "",
                "alpha_model_path": "",
                "scenario": scenario_by_id.get(scenario_id, ""),
                "scenario_id": str(scenario_id),
                "seed": str(seed),
                "steps": "",
                "noise_scale": "",
                "risk_gain": "",
                "min_distance_scale": "",
                "status": "file_only" if ego_log.exists() and nbr_log.exists() else "missing",
                "ego_log": str(ego_log),
                "nbr_log": str(nbr_log),
                "carla_log": "",
            }
        target_rows.append(row)

    completeness = []
    for planner in PLANNERS:
        sigma_iter = ["estimated"] if planner == "B1" else SIGMA_SOURCES
        for sigma in sigma_iter:
            exp = len(scenarios) * len(seeds)
            missing_files = 0
            found = 0
            for row in target_rows:
                if row["planner"] != planner or row["sigma_source"] != sigma:
                    continue
                if not Path(row["ego_log"]).exists() or not Path(row["nbr_log"]).exists():
                    missing_files += 1
                else:
                    found += 1
            completeness.append(
                {
                    "planner": planner,
                    "sigma_source": sigma,
                    "expected_runs": exp,
                    "found_runs": found,
                    "missing_runs": exp - found,
                    "missing_files": missing_files,
                }
            )

    run_rows: list[dict[str, object]] = []
    missing_file_rows = []
    for n, row in enumerate(sorted(target_rows, key=lambda r: (i(r["scenario_id"]), r["planner"], r["sigma_source"], i(r["seed"]))), 1):
        ego_path = Path(row["ego_log"])
        nbr_path = Path(row["nbr_log"])
        if not ego_path.exists() or not nbr_path.exists():
            missing_file_rows.append(row)
            continue
        summary = {
            "run_id": row["run_id"],
            "planner": row["planner"],
            "sigma_source": row["sigma_source"],
            "scenario_id": i(row["scenario_id"]),
            "scenario": row["scenario"],
            "seed": i(row["seed"]),
            "noise_scale": f(row["noise_scale"]),
        }
        summary.update(summarize_ego(ego_path))
        nbr_summary = summarize_nbr(nbr_path)
        if row["noise_scale"] == "":
            summary["noise_scale"] = nbr_summary["mean_noise_scale_nbr"]
        summary.update(nbr_summary)
        run_rows.append(summary)
        if n % 250 == 0:
            print(f"[INFO] summarized {n}/{len(target_rows)} manifest runs")

    run_fields = [
        "run_id",
        "planner",
        "sigma_source",
        "scenario_id",
        "scenario",
        "seed",
        "noise_scale",
        "mean_noise_scale_nbr",
        "ego_steps",
        "nbr_rows",
        "mean_speed",
        "path_distance",
        "mean_throttle",
        "mean_brake",
        "max_brake",
        "brake_step_frac",
        "mean_planner_risk",
        "max_planner_risk",
        "min_front_distance",
        "min_effective_front_distance",
        "collision_run",
        "near_miss_true_run",
        "near_miss_obs_run",
        "min_ttc_true",
        "min_ttc_obs",
        "mean_planner_sigma_nbr",
        "max_planner_sigma_nbr",
        "mean_alpha_nbr",
        "max_alpha_nbr",
        "mean_risk_nbr",
        "max_risk_nbr",
        "mean_position_error",
        "mean_velocity_error",
        "slot_rows",
        "mean_slot_sigma",
        "mean_slot_risk",
    ]
    metric_fields = [field for field in run_fields if field not in {"run_id", "planner", "sigma_source", "scenario", "scenario_id", "seed"}]
    metric_fields = [field for field in metric_fields if field != "noise_scale"]

    group_rows = numeric_summary(run_rows, ("planner", "sigma_source"), metric_fields)
    scenario_group_rows = numeric_summary(run_rows, ("scenario_id", "planner", "sigma_source"), metric_fields)
    noise_group_rows = numeric_summary(run_rows, ("noise_scale", "planner", "sigma_source"), metric_fields)
    pair_rows = build_pairwise(run_rows, "B3", "B2")
    pair_group_rows = numeric_summary(
        pair_rows,
        ("sigma_source",),
        [field for field in pair_rows[0].keys() if field.endswith("_diff")] if pair_rows else [],
    )

    write_csv(output_dir / "run_summary.csv", run_rows, run_fields)
    write_csv(output_dir / "completeness.csv", completeness, list(completeness[0].keys()))
    write_csv(output_dir / "planner_sigma_summary.csv", group_rows, list(group_rows[0].keys()))
    write_csv(output_dir / "scenario_planner_sigma_summary.csv", scenario_group_rows, list(scenario_group_rows[0].keys()))
    write_csv(output_dir / "noise_planner_sigma_summary.csv", noise_group_rows, list(noise_group_rows[0].keys()))
    if pair_rows:
        write_csv(output_dir / "b3_minus_b2_pairs.csv", pair_rows, list(pair_rows[0].keys()))
        write_csv(output_dir / "b3_minus_b2_summary.csv", pair_group_rows, list(pair_group_rows[0].keys()))
    make_markdown(output_dir / "analysis_report.md", completeness, group_rows, pair_rows)

    print(f"[INFO] wrote {len(run_rows)} run summaries to {output_dir}")
    if missing_file_rows:
        print(f"[WARN] skipped {len(missing_file_rows)} manifest rows with missing files")


if __name__ == "__main__":
    main()
