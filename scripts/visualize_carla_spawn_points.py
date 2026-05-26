#!/usr/bin/env python3
"""Visualize CARLA spawn points and lane markings in a top-down SVG map."""

from __future__ import annotations

import argparse
import csv
import glob
import html
import math
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
CARLA_ROOT = REPO_ROOT / "carla-simulator"


def add_carla_to_path() -> None:
    """Prefer the bundled CARLA PythonAPI when it is available."""
    python_api = CARLA_ROOT / "PythonAPI"
    sys.path.append(str(python_api / "carla"))

    version = f"cp{sys.version_info.major}{sys.version_info.minor}"
    wheel_pattern = str(python_api / "carla" / "dist" / f"carla-*{version}*.whl")
    for wheel in glob.glob(wheel_pattern):
        sys.path.append(wheel)
        return


carla = None


def import_carla() -> None:
    global carla

    add_carla_to_path()
    try:
        import carla as carla_module
    except ImportError as exc:  # pragma: no cover - depends on local CARLA install.
        version = f"{sys.version_info.major}.{sys.version_info.minor}"
        available = sorted(path.name for path in (CARLA_ROOT / "PythonAPI" / "carla" / "dist").glob("carla-*.whl"))
        available_text = ", ".join(available) if available else "none found"
        raise SystemExit(
            f"Could not import CARLA PythonAPI for Python {version}.\n"
            f"Bundled wheels: {available_text}\n"
            "Use a Python version with a matching wheel, e.g. python3.12 or python3.11 "
            "for this CARLA bundle, or install a matching CARLA Python package."
        ) from exc
    carla = carla_module


Point = Tuple[float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a top-down SVG/CSV showing CARLA map spawn point ids, "
            "locations, headings, lanes, and lane markings."
        )
    )
    parser.add_argument("--host", default="127.0.0.1", help="CARLA host.")
    parser.add_argument("--port", default=2000, type=int, help="CARLA RPC port.")
    parser.add_argument("--timeout", default=20.0, type=float, help="Client timeout in seconds.")
    parser.add_argument("--map", default="Town04", help="Map to load/use, e.g. Town04.")
    parser.add_argument(
        "--no-load-map",
        action="store_true",
        help="Do not call client.load_world(); use the currently loaded map.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/map_visualizations",
        help="Directory for the generated SVG and CSV files.",
    )
    parser.add_argument(
        "--sampling-distance",
        default=2.0,
        type=float,
        help="Waypoint sampling distance in meters for roads/markings.",
    )
    parser.add_argument(
        "--padding",
        default=20.0,
        type=float,
        help="Map padding in meters around all sampled geometry.",
    )
    parser.add_argument(
        "--max-labels",
        default=10000,
        type=int,
        help="Maximum spawn point labels to draw. CSV always contains all spawn points.",
    )
    return parser.parse_args()


def map_name(carla_map: "carla.Map") -> str:
    return carla_map.name.rsplit("/", 1)[-1]


def connect(args: argparse.Namespace) -> Tuple["carla.Client", "carla.World", "carla.Map"]:
    import_carla()
    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)

    world = client.get_world()
    current_map = world.get_map()
    if not args.no_load_map and map_name(current_map) != args.map:
        world = client.load_world(args.map)
        current_map = world.get_map()

    return client, world, current_map


def yaw_vector(yaw_degrees: float, length: float = 1.0) -> Point:
    yaw = math.radians(yaw_degrees)
    return math.cos(yaw) * length, math.sin(yaw) * length


def offset_location(location: "carla.Location", yaw_degrees: float, lateral_offset: float) -> Point:
    yaw = math.radians(yaw_degrees)
    # Left normal of the lane heading.
    return (
        location.x - math.sin(yaw) * lateral_offset,
        location.y + math.cos(yaw) * lateral_offset,
    )


def point_from_location(location: "carla.Location") -> Point:
    return location.x, location.y


def svg_point(point: Point, min_x: float, max_y: float, padding: float) -> Point:
    x, y = point
    return x - min_x + padding, max_y - y + padding


def svg_polyline(points: Sequence[Point], min_x: float, max_y: float, padding: float) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in (svg_point(p, min_x, max_y, padding) for p in points))


def lane_marking_color(marking: "carla.LaneMarking") -> str:
    color = str(getattr(marking, "color", "")).lower()
    if "yellow" in color:
        return "#d7a400"
    if "blue" in color:
        return "#2d6cdf"
    if "green" in color:
        return "#1c9b62"
    if "red" in color:
        return "#c94141"
    return "#f5f5f2"


def lane_marking_dash(marking: "carla.LaneMarking") -> str:
    marking_type = str(getattr(marking, "type", "")).lower()
    if "broken" in marking_type:
        return ' stroke-dasharray="4 5"'
    if "botts" in marking_type:
        return ' stroke-dasharray="1 5"'
    if "curb" in marking_type:
        return ' stroke-dasharray="8 3 1 3"'
    return ""


def marking_width(marking: "carla.LaneMarking") -> float:
    width = float(getattr(marking, "width", 0.15) or 0.15)
    return max(width, 0.08)


def is_visible_marking(marking: "carla.LaneMarking") -> bool:
    marking_type = str(getattr(marking, "type", "")).lower()
    return "none" not in marking_type and "unknown" not in marking_type


def collect_waypoint_segments(
    carla_map: "carla.Map",
    sampling_distance: float,
) -> Tuple[List[Tuple["carla.Waypoint", "carla.Waypoint"]], List[Point]]:
    waypoints = carla_map.generate_waypoints(sampling_distance)
    segments = []
    all_points = []

    for waypoint in waypoints:
        next_waypoints = waypoint.next(sampling_distance)
        for next_waypoint in next_waypoints:
            # Keep junction fan-out, but avoid huge accidental jumps.
            distance = waypoint.transform.location.distance(next_waypoint.transform.location)
            if distance > sampling_distance * 2.5:
                continue
            segments.append((waypoint, next_waypoint))
            all_points.append(point_from_location(waypoint.transform.location))
            all_points.append(point_from_location(next_waypoint.transform.location))

    return segments, all_points


def bounds(points: Iterable[Point]) -> Tuple[float, float, float, float]:
    points = list(points)
    if not points:
        return -10.0, -10.0, 10.0, 10.0
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def waypoint_for_spawn(carla_map: "carla.Map", transform: "carla.Transform") -> Optional["carla.Waypoint"]:
    return carla_map.get_waypoint(
        transform.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )


def write_spawn_csv(
    csv_path: Path,
    carla_map: "carla.Map",
    spawn_points: Sequence["carla.Transform"],
) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "id",
                "x",
                "y",
                "z",
                "pitch",
                "yaw",
                "roll",
                "road_id",
                "lane_id",
                "section_id",
                "junction_id",
                "is_junction",
            ]
        )
        for spawn_id, transform in enumerate(spawn_points):
            waypoint = waypoint_for_spawn(carla_map, transform)
            writer.writerow(
                [
                    spawn_id,
                    f"{transform.location.x:.6f}",
                    f"{transform.location.y:.6f}",
                    f"{transform.location.z:.6f}",
                    f"{transform.rotation.pitch:.6f}",
                    f"{transform.rotation.yaw:.6f}",
                    f"{transform.rotation.roll:.6f}",
                    getattr(waypoint, "road_id", ""),
                    getattr(waypoint, "lane_id", ""),
                    getattr(waypoint, "section_id", ""),
                    getattr(waypoint, "junction_id", ""),
                    getattr(waypoint, "is_junction", ""),
                ]
            )


def build_svg(
    town: str,
    spawn_points: Sequence["carla.Transform"],
    segments: Sequence[Tuple["carla.Waypoint", "carla.Waypoint"]],
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
    padding: float,
    max_labels: int,
) -> str:
    width = max_x - min_x + padding * 2.0
    height = max_y - min_y + padding * 2.0
    road_lines = []
    center_lines = []
    marking_lines = []
    spawn_markup = []

    for waypoint, next_waypoint in segments:
        p1 = point_from_location(waypoint.transform.location)
        p2 = point_from_location(next_waypoint.transform.location)
        center_points = svg_polyline([p1, p2], min_x, max_y, padding)
        lane_width = max(float(getattr(waypoint, "lane_width", 3.5) or 3.5), 1.5)
        road_lines.append(
            f'<polyline points="{center_points}" class="road" stroke-width="{lane_width:.2f}" />'
        )
        center_lines.append(f'<polyline points="{center_points}" class="centerline" />')

        for side, lateral_offset, attr in (
            ("left", lane_width / 2.0, "left_lane_marking"),
            ("right", -lane_width / 2.0, "right_lane_marking"),
        ):
            marking = getattr(waypoint, attr, None)
            if marking is None or not is_visible_marking(marking):
                continue
            start = offset_location(waypoint.transform.location, waypoint.transform.rotation.yaw, lateral_offset)
            end = offset_location(next_waypoint.transform.location, next_waypoint.transform.rotation.yaw, lateral_offset)
            points = svg_polyline([start, end], min_x, max_y, padding)
            marking_lines.append(
                '<polyline '
                f'points="{points}" '
                f'class="lane-marking {side}" '
                f'stroke="{lane_marking_color(marking)}" '
                f'stroke-width="{marking_width(marking):.2f}"'
                f'{lane_marking_dash(marking)} />'
            )

    for spawn_id, transform in enumerate(spawn_points[:max_labels]):
        loc = transform.location
        x, y = svg_point((loc.x, loc.y), min_x, max_y, padding)
        dx, dy = yaw_vector(transform.rotation.yaw, length=5.0)
        arrow_end = svg_point((loc.x + dx, loc.y + dy), min_x, max_y, padding)
        tooltip = html.escape(
            f"id={spawn_id}, x={loc.x:.2f}, y={loc.y:.2f}, z={loc.z:.2f}, yaw={transform.rotation.yaw:.1f}"
        )
        spawn_markup.append(
            f'<g class="spawn"><title>{tooltip}</title>'
            f'<line x1="{x:.2f}" y1="{y:.2f}" x2="{arrow_end[0]:.2f}" y2="{arrow_end[1]:.2f}" />'
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="1.7" />'
            f'<text x="{x + 2.2:.2f}" y="{y - 2.2:.2f}">{spawn_id}</text>'
            "</g>"
        )

    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     width="1800"
     height="{max(900, int(1800 * height / max(width, 1.0)))}"
     viewBox="0 0 {width:.2f} {height:.2f}"
     role="img"
     aria-label="{html.escape(town)} CARLA spawn point map">
  <defs>
    <marker id="spawn-arrow" viewBox="0 0 10 10" refX="8" refY="5"
            markerWidth="3" markerHeight="3" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#d92332" />
    </marker>
    <style>
      .background {{ fill: #151719; }}
      .road {{ fill: none; stroke: #555b61; stroke-linecap: round; stroke-linejoin: round; opacity: 0.88; }}
      .centerline {{ fill: none; stroke: #1f2327; stroke-width: 0.20; stroke-linecap: round; opacity: 0.35; }}
      .lane-marking {{ fill: none; stroke-linecap: round; stroke-linejoin: round; opacity: 0.96; }}
      .spawn line {{ stroke: #d92332; stroke-width: 0.55; marker-end: url(#spawn-arrow); }}
      .spawn circle {{ fill: #ff3948; stroke: #ffffff; stroke-width: 0.45; }}
      .spawn text {{
        fill: #ffffff;
        stroke: #101214;
        stroke-width: 0.45;
        paint-order: stroke;
        font-family: Arial, Helvetica, sans-serif;
        font-size: 3.2px;
        font-weight: 700;
      }}
      .title {{
        fill: #ffffff;
        font-family: Arial, Helvetica, sans-serif;
        font-size: 7px;
        font-weight: 700;
      }}
      .subtitle {{
        fill: #c9ced3;
        font-family: Arial, Helvetica, sans-serif;
        font-size: 3.3px;
      }}
    </style>
  </defs>
  <rect class="background" width="100%" height="100%" />
  <g id="roads">
    {os.linesep.join(road_lines)}
  </g>
  <g id="lane-centerlines">
    {os.linesep.join(center_lines)}
  </g>
  <g id="lane-markings">
    {os.linesep.join(marking_lines)}
  </g>
  <g id="spawn-points">
    {os.linesep.join(spawn_markup)}
  </g>
  <text x="6" y="11" class="title">{html.escape(town)} spawn points</text>
  <text x="6" y="17" class="subtitle">red ids: map.get_spawn_points() order, white/yellow lines: sampled lane markings</text>
</svg>
'''


def main() -> None:
    args = parse_args()
    _, _, carla_map = connect(args)
    town = map_name(carla_map)
    spawn_points = carla_map.get_spawn_points()
    if not spawn_points:
        raise SystemExit(f"{town} has no vehicle spawn points.")

    segments, geometry_points = collect_waypoint_segments(carla_map, args.sampling_distance)
    geometry_points.extend(point_from_location(spawn.location) for spawn in spawn_points)
    min_x, min_y, max_x, max_y = bounds(geometry_points)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{town}_spawn_points"
    csv_path = output_dir / f"{stem}.csv"
    svg_path = output_dir / f"{stem}.svg"

    write_spawn_csv(csv_path, carla_map, spawn_points)
    svg = build_svg(
        town=town,
        spawn_points=spawn_points,
        segments=segments,
        min_x=min_x,
        min_y=min_y,
        max_x=max_x,
        max_y=max_y,
        padding=args.padding,
        max_labels=args.max_labels,
    )
    svg_path.write_text(svg, encoding="utf-8")

    print(f"Map: {town}")
    print(f"Spawn points: {len(spawn_points)}")
    print(f"Waypoint segments: {len(segments)}")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {svg_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled.")
