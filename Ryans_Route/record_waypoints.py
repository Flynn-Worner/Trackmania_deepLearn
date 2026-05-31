"""
Waypoint Recorder — drive the ideal racing line with telemetry.as active.

Usage:
    python record_waypoints.py                   # default port 8483, 10 m spacing
    python record_waypoints.py --port 8483 --spacing 5
    python record_waypoints.py --draft           # save to data/waypoints_draft.json
    python record_waypoints.py --output data/my_waypoints.json

Workflow:
    1. Start TM with telemetry.as plugin enabled.
  2. Load the map you want to train on.
    3. Run this script and drive the center racing line.
  4. Drive your car down the MIDDLE of the track at your own pace.
    5. Press Ctrl+C when finished; data/waypoints.json is written.

After recording, run:
    python generate_gates.py
to produce data/gates.json for training.
"""

import argparse
import json
import math
import os
import signal
import time

from telemetry_bridge import TelemetryBridge


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SPACING_M = 10.0    # minimum metres between recorded waypoints
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "waypoints.json")
DRAFT_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "waypoints_draft.json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dist3d(a, b):
    """Euclidean distance between two (x, y, z) sequences."""
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def _extract_yaw(state):
    """
    Extract yaw from telemetry bridge state.
    """
    return float(state.yaw)


# ---------------------------------------------------------------------------
# Main recorder
# ---------------------------------------------------------------------------

def record(port: int, spacing_m: float, output_file: str = OUTPUT_FILE,
           draft: bool = False):
    bridge = TelemetryBridge(port=port)
    print(f"[Recorder] Waiting for telemetry.as connection on port {port} ...")
    try:
        bridge.connect()
    except TimeoutError:
        print(
            "[Recorder] Timed out waiting for telemetry.as connection. "
            "Start the game/plugin first, then run the recorder again."
        )
        return
    bridge.send_action("MANUAL")

    print("[Recorder] Connected to telemetry.as")
    print(f"[Recorder] Drive the car down the MIDDLE of the track.")
    print(f"[Recorder] Waypoints captured every {spacing_m:.1f} m of movement.")
    print(f"[Recorder] Press Ctrl+C to stop recording early.\n")

    waypoints = []
    last_pos = None
    running = True

    def _sigint_handler(sig, frame):
        nonlocal running
        print("\n[Recorder] Ctrl+C received — stopping recording ...")
        running = False

    signal.signal(signal.SIGINT, _sigint_handler)

    try:
        while running:
            state = bridge.recv_state()
            bridge.send_action("MANUAL")
            race_time = state.sample_idx * 100

            pos = list(state.position)       # [x, y, z]
            yaw = float(_extract_yaw(state))
            speed = state.display_speed

            # Only record if we've moved far enough from the last waypoint
            if last_pos is None or _dist3d(pos, last_pos) >= spacing_m:
                wp = {
                    "time_ms": race_time,
                    "position": pos,
                    "yaw": yaw,
                    "speed_kmh": speed,
                }
                waypoints.append(wp)
                last_pos = pos

                n = len(waypoints)
                if n % 10 == 0 or n == 1:
                    print(
                        f"  [{n:4d}] t={race_time:6d}ms  "
                        f"pos=({pos[0]:8.1f}, {pos[1]:6.1f}, {pos[2]:8.1f})  "
                        f"yaw={math.degrees(yaw):6.1f}°  "
                        f"speed={speed:5.1f} km/h"
                    )

    except Exception as e:
        print(f"\n[Recorder] Error: {e}")
    finally:
        # Leave plugin in MANUAL control when recorder exits.
        try:
            bridge.send_action("MANUAL")
        except Exception:
            pass
        bridge.close()

    # Save
    if not waypoints:
        print("[Recorder] No waypoints recorded — nothing to save.")
        return

    out_path = os.path.abspath(output_file)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    output = {
        "version": 1,
        "spacing_m": spacing_m,
        "draft": bool(draft),
        "finished": False,
        "num_waypoints": len(waypoints),
        "waypoints": waypoints,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\n[Recorder] Saved {len(waypoints)} waypoints → {out_path}")
    if draft:
        print("[Recorder] Draft recording complete (safe to delete later).")
    print(f"[Recorder] Next step: python generate_gates.py")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Record the ideal racing line as waypoints",
    )
    parser.add_argument("--port", type=int, default=9000,
                        help="telemetry.as socket port (default: 9000)")
    parser.add_argument("--spacing", type=float, default=DEFAULT_SPACING_M,
                        help=f"Min metres between waypoints (default: {DEFAULT_SPACING_M})")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for waypoints JSON (default: data/waypoints.json)")
    parser.add_argument("--draft", action="store_true",
                        help="Save to data/waypoints_draft.json instead of overwriting waypoints.json")
    args = parser.parse_args()

    output_file = args.output
    if output_file is None:
        output_file = DRAFT_OUTPUT_FILE if args.draft else OUTPUT_FILE

    record(args.port, args.spacing, output_file=output_file, draft=args.draft)


if __name__ == "__main__":
    main()
