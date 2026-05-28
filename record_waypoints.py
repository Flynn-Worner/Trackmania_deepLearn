"""
Waypoint Recorder — drive the ideal racing line at 0.1× speed.

Usage:
    python record_waypoints.py                   # default port 8483, 10 m spacing
    python record_waypoints.py --port 8483 --spacing 5

Workflow:
  1. Open TmForever + TMInterface on the chosen port.
  2. Load the map you want to train on.
  3. Run this script — it sets the game to 0.1× speed.
  4. Drive your car down the MIDDLE of the track at your own pace.
  5. When you cross the finish line (or press Ctrl+C), recording stops
     and data/waypoints.json is written.

After recording, run:
    python generate_gates.py
to produce data/gates.json for training.
"""

import argparse
import json
import math
import os
import signal
import struct
import sys
import time

# ---------------------------------------------------------------------------
# We reuse the existing TMInterface client so we don't duplicate socket code.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from interfacing.tminterface2 import TMInterface, MessageType


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RECORD_SPEED = 0.1          # game speed multiplier during recording
DEFAULT_SPACING_M = 10.0    # minimum metres between recorded waypoints
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "waypoints.json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dist3d(a, b):
    """Euclidean distance between two (x, y, z) sequences."""
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def _extract_yaw(state):
    """
    Extract the car's yaw angle from the simulation state.
    TMInterface gives yaw_pitch_roll as a tuple (yaw, pitch, roll) in radians.
    """
    yaw, _pitch, _roll = state.yaw_pitch_roll
    return yaw


# ---------------------------------------------------------------------------
# Main recorder
# ---------------------------------------------------------------------------

def record(port: int, spacing_m: float):
    iface = TMInterface(port)
    print(f"[Recorder] Connecting to TMInterface on port {port} ...")
    iface.register(timeout=None)

    # Wait for on_connect
    while True:
        msgtype = iface._read_int32()
        if msgtype == int(MessageType.SC_ON_CONNECT_SYNC):
            # Set up: slow game speed, step period = 10 ticks (100 ms game time)
            iface.set_on_step_period(10)
            iface.set_speed(RECORD_SPEED)
            iface._respond_to_call(msgtype)
            break
        else:
            iface._respond_to_call(msgtype)

    print(f"[Recorder] Connected — game speed set to {RECORD_SPEED}×")
    print(f"[Recorder] Drive the car down the MIDDLE of the track.")
    print(f"[Recorder] Waypoints captured every {spacing_m:.1f} m of movement.")
    print(f"[Recorder] Press Ctrl+C to stop recording early.\n")

    waypoints = []
    last_pos = None
    running = True
    finished = False

    def _sigint_handler(sig, frame):
        nonlocal running
        print("\n[Recorder] Ctrl+C received — stopping recording ...")
        running = False

    signal.signal(signal.SIGINT, _sigint_handler)

    try:
        while running:
            msgtype = iface._read_int32()

            if msgtype == int(MessageType.SC_RUN_STEP_SYNC):
                race_time = iface._read_int32()
                state = iface.get_simulation_state()
                iface._respond_to_call(msgtype)

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

            elif msgtype == int(MessageType.SC_CHECKPOINT_COUNT_CHANGED_SYNC):
                current = iface._read_int32()
                target = iface._read_int32()
                iface._respond_to_call(msgtype)

                if current == target:
                    print(f"\n[Recorder] FINISH LINE crossed — stopping recording.")
                    finished = True
                    running = False

            else:
                iface._respond_to_call(msgtype)

    except Exception as e:
        print(f"\n[Recorder] Error: {e}")
    finally:
        # Restore normal speed so the game isn't stuck at 0.1×
        try:
            iface.set_speed(1.0)
        except Exception:
            pass

    # Save
    if not waypoints:
        print("[Recorder] No waypoints recorded — nothing to save.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output = {
        "version": 1,
        "spacing_m": spacing_m,
        "finished": finished,
        "num_waypoints": len(waypoints),
        "waypoints": waypoints,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n[Recorder] Saved {len(waypoints)} waypoints → {OUTPUT_FILE}")
    print(f"[Recorder] Next step: python generate_gates.py")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Record the ideal racing line as waypoints",
    )
    parser.add_argument("--port", type=int, default=8483,
                        help="TMInterface port (default: 8483)")
    parser.add_argument("--spacing", type=float, default=DEFAULT_SPACING_M,
                        help=f"Min metres between waypoints (default: {DEFAULT_SPACING_M})")
    args = parser.parse_args()
    record(args.port, args.spacing)


if __name__ == "__main__":
    main()
