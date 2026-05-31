"""
Gate Generator — convert recorded waypoints into training gates.

Usage:
    python generate_gates.py                     # defaults
    python generate_gates.py --width 32          # override track width
    python generate_gates.py --visualize         # print gate summary

Reads:  data/waypoints.json   (output of record_waypoints.py)
Writes: data/gates.json       (consumed by env.py for training)

Each gate is a line segment across the track at each waypoint.
If wall points are available, orientation comes from left/right wall geometry.
Otherwise it falls back to waypoint yaw.

Track width: TMNF Stadium blocks are 32 m wide.
"""

import argparse
import json
import math
import os
import sys

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TRACK_WIDTH = 32.0   # metres — standard TMNF stadium track width
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
INPUT_FILE = os.path.join(DATA_DIR, "waypoints.json")
OUTPUT_FILE = os.path.join(DATA_DIR, "gates.json")
WALLS_FILE = os.path.join(DATA_DIR, "wall_points.json")
LEFT_WALL_FILE = os.path.join(DATA_DIR, "left_wall_points.json")
RIGHT_WALL_FILE = os.path.join(DATA_DIR, "right_wall_points.json")


# ---------------------------------------------------------------------------
# Gate math
# ---------------------------------------------------------------------------

def _yaw_to_forward_xz(yaw: float):
    """
    Convert TMInterface yaw to a unit forward vector in the XZ plane.

    In TMNF / TMInterface:
      yaw = 0   → car faces +Z
      yaw = π/2 → car faces -X  (left turn from +Z)

    Forward vector: (-sin(yaw), cos(yaw)) in the XZ plane.
    """
    fx = -math.sin(yaw)
    fz = math.cos(yaw)
    return fx, fz


def _perpendicular_xz(fx, fz):
    """
    Return the unit vector perpendicular to (fx, fz) in the XZ plane.
    Rotation +90° → (fz, -fx).   This gives the "right" direction
    when the car faces forward.
    """
    return fz, -fx


def _load_wall_points() -> tuple[np.ndarray, np.ndarray] | None:
    """Load left/right wall XZ points if available."""
    if os.path.exists(WALLS_FILE):
        with open(WALLS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        left = np.asarray(data.get("left_wall_points_xz", []), dtype=np.float64)
        right = np.asarray(data.get("right_wall_points_xz", []), dtype=np.float64)
        if len(left) > 1 and len(right) > 1:
            return left, right

    if os.path.exists(LEFT_WALL_FILE) and os.path.exists(RIGHT_WALL_FILE):
        with open(LEFT_WALL_FILE, "r", encoding="utf-8") as f:
            left_data = json.load(f)
        with open(RIGHT_WALL_FILE, "r", encoding="utf-8") as f:
            right_data = json.load(f)

        left = np.asarray(left_data.get("points_xz", []), dtype=np.float64)
        right = np.asarray(right_data.get("points_xz", []), dtype=np.float64)
        if len(left) > 1 and len(right) > 1:
            return left, right

    return None


def _wall_oriented_axes(cx: float, cz: float, yaw: float,
                        left_wall: np.ndarray, right_wall: np.ndarray):
    """Return (forward_xz, across_xz) using nearest left/right wall points.

    across points from left wall to right wall.
    forward is perpendicular to across and aligned with yaw direction.
    """
    center = np.array([cx, cz], dtype=np.float64)

    li = int(np.argmin(np.sum((left_wall - center) ** 2, axis=1)))
    ri = int(np.argmin(np.sum((right_wall - center) ** 2, axis=1)))

    span = right_wall[ri] - left_wall[li]
    span_norm = float(np.linalg.norm(span))
    if span_norm <= 1e-6:
        return None

    across = span / span_norm
    # forward = perpendicular to across
    fx, fz = -float(across[1]), float(across[0])

    yaw_fx, yaw_fz = _yaw_to_forward_xz(yaw)
    if fx * yaw_fx + fz * yaw_fz < 0.0:
        fx, fz = -fx, -fz

    return (fx, fz), (float(across[0]), float(across[1]))


def generate_gates(waypoints: list, track_width: float,
                   wall_points: tuple[np.ndarray, np.ndarray] | None = None) -> list:
    """
    For each waypoint, build a gate: two posts at ±(track_width/2)
    along the perpendicular to the car's heading.

    Returns a list of gate dicts.
    """
    half_w = track_width / 2.0
    gates = []

    left_wall = None
    right_wall = None
    if wall_points is not None:
        left_wall, right_wall = wall_points

    for i, wp in enumerate(waypoints):
        pos = wp["position"]       # [x, y, z]
        yaw = wp["yaw"]
        speed = wp.get("speed_kmh", 0.0)

        cx, cy, cz = pos[0], pos[1], pos[2]

        if left_wall is not None and right_wall is not None:
            axes = _wall_oriented_axes(float(cx), float(cz), float(yaw), left_wall, right_wall)
        else:
            axes = None

        if axes is not None:
            (fx, fz), (px, pz) = axes
        else:
            fx, fz = _yaw_to_forward_xz(yaw)
            px, pz = _perpendicular_xz(fx, fz)

        # across axis points left->right, so posts are symmetric around center
        left = [cx - px * half_w, cy, cz - pz * half_w]
        right = [cx + px * half_w, cy, cz + pz * half_w]

        # Forward direction for 2D intersection math later
        gate = {
            "gate_id": i,
            "center": [cx, cy, cz],
            "left_post": left,
            "right_post": right,
            "forward": [fx, fz],
            "target_speed": speed,
        }
        gates.append(gate)

    return gates


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert waypoints to gates")
    parser.add_argument("--width", type=float, default=DEFAULT_TRACK_WIDTH,
                        help=f"Track width in metres (default: {DEFAULT_TRACK_WIDTH})")
    parser.add_argument("--input", type=str, default=INPUT_FILE,
                        help=f"Waypoints input file (default: {INPUT_FILE})")
    parser.add_argument("--output", type=str, default=OUTPUT_FILE,
                        help=f"Gates output file (default: {OUTPUT_FILE})")
    parser.add_argument("--visualize", action="store_true",
                        help="Print gate positions after generation")
    parser.add_argument("--no-walls", action="store_true",
                        help="Ignore wall points and use yaw-only orientation")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: {args.input} not found.")
        print("Run record_waypoints.py first to record the racing line.")
        sys.exit(1)

    with open(args.input, "r") as f:
        data = json.load(f)

    wp_list = data.get("waypoints", data) if isinstance(data, dict) else data
    if not wp_list:
        print("ERROR: No waypoints found in input file.")
        sys.exit(1)

    print(f"[GateGen] Loaded {len(wp_list)} waypoints from {args.input}")
    print(f"[GateGen] Track width: {args.width:.1f} m")

    wall_points = None
    if not args.no_walls:
        wall_points = _load_wall_points()
        if wall_points is not None:
            print(
                f"[GateGen] Using wall-based orientation "
                f"(left={len(wall_points[0])} pts, right={len(wall_points[1])} pts)"
            )
        else:
            print("[GateGen] Wall points not found; falling back to yaw-based orientation")
    else:
        print("[GateGen] Wall orientation disabled via --no-walls")

    gates = generate_gates(wp_list, args.width, wall_points=wall_points)

    # Compute arc distances between consecutive gates for metadata
    total_arc = 0.0
    for i in range(1, len(gates)):
        c0 = gates[i - 1]["center"]
        c1 = gates[i]["center"]
        d = math.sqrt(sum((a - b) ** 2 for a, b in zip(c0, c1)))
        total_arc += d

    output = {
        "version": 1,
        "track_width_m": args.width,
        "num_gates": len(gates),
        "total_arc_m": round(total_arc, 2),
        "gates": gates,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"[GateGen] Generated {len(gates)} gates → {args.output}")
    print(f"[GateGen] Total arc length: {total_arc:.1f} m")

    if args.visualize:
        print(f"\n{'='*70}")
        print(f"  Gate  |  Center (x,y,z)              | Target Speed")
        print(f"{'='*70}")
        for g in gates:
            c = g["center"]
            print(f"  {g['gate_id']:4d}  | "
                  f"({c[0]:8.1f}, {c[1]:6.1f}, {c[2]:8.1f}) | "
                  f"{g['target_speed']:6.1f} km/h")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
