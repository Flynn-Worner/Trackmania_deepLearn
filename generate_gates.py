"""
Gate Generator — convert recorded waypoints into training gates.

Usage:
    python generate_gates.py                     # defaults
    python generate_gates.py --width 32          # override track width
    python generate_gates.py --visualize         # print gate summary

Reads:  data/waypoints.json   (output of record_waypoints.py)
Writes: data/gates.json       (consumed by env.py for training)

Each gate is a line segment perpendicular to the car's heading at that
waypoint, spanning TRACK_WIDTH/2 in each direction.  The RL agent is
rewarded for crossing gates in order.

Track width: TMNF Stadium blocks are 32 m wide.
"""

import argparse
import json
import math
import os
import sys


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TRACK_WIDTH = 32.0   # metres — standard TMNF stadium track width
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
INPUT_FILE = os.path.join(DATA_DIR, "waypoints.json")
OUTPUT_FILE = os.path.join(DATA_DIR, "gates.json")


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


def generate_gates(waypoints: list, track_width: float) -> list:
    """
    For each waypoint, build a gate: two posts at ±(track_width/2)
    along the perpendicular to the car's heading.

    Returns a list of gate dicts.
    """
    half_w = track_width / 2.0
    gates = []

    for i, wp in enumerate(waypoints):
        pos = wp["position"]       # [x, y, z]
        yaw = wp["yaw"]
        speed = wp.get("speed_kmh", 0.0)

        fx, fz = _yaw_to_forward_xz(yaw)
        px, pz = _perpendicular_xz(fx, fz)

        cx, cy, cz = pos[0], pos[1], pos[2]

        # Left post (perpendicular * +half_w)
        left = [cx + px * half_w, cy, cz + pz * half_w]
        # Right post (perpendicular * -half_w)
        right = [cx - px * half_w, cy, cz - pz * half_w]

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

    gates = generate_gates(wp_list, args.width)

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
