"""Manual gate monitor for Trackmania telemetry.as.

Run this while driving the car yourself. It listens to live telemetry,
checks the same gate geometry used by training, and prints each gate
as it is crossed.

Usage:
    python manual_gate_test.py --port 9000
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass

from telemetry_bridge import TelemetryBridge
from rewards import segments_intersect_2d


@dataclass
class GateCheckConfig:
    gate_y_tolerance: float = 7.5


def _load_gates() -> list[dict]:
    gates_path = os.path.join(os.path.dirname(__file__), "data", "gates.json")
    if not os.path.exists(gates_path):
        raise FileNotFoundError(f"Missing gates file: {gates_path}")

    with open(gates_path, "r", encoding="utf-8") as handle:
        gate_data = json.load(handle)

    gates = gate_data.get("gates", gate_data) if isinstance(gate_data, dict) else gate_data
    if not gates:
        raise ValueError("No gates found in data/gates.json")
    return gates


def _check_gate_crossing(prev_pos, cur_pos, gate: dict, cfg: GateCheckConfig) -> bool:
    center = gate["center"]
    if abs(float(cur_pos[1]) - float(center[1])) > cfg.gate_y_tolerance:
        return False

    left = gate["left_post"]
    right = gate["right_post"]

    p1 = (float(prev_pos[0]), float(prev_pos[2]))
    p2 = (float(cur_pos[0]), float(cur_pos[2]))
    q1 = (float(left[0]), float(left[2]))
    q2 = (float(right[0]), float(right[2]))
    return segments_intersect_2d(p1, p2, q1, q2)


def _dist_to_gate_center(pos, gate: dict) -> float:
    center = gate["center"]
    dx = float(pos[0]) - float(center[0])
    dy = float(pos[1]) - float(center[1])
    dz = float(pos[2]) - float(center[2])
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def monitor(port: int):
    gates = _load_gates()
    cfg = GateCheckConfig()

    bridge = TelemetryBridge(port=port)
    print(f"[GateTest] Waiting for telemetry.as connection on port {port} ...")
    bridge.connect()
    bridge.send_action("MANUAL")
    print(f"[GateTest] Connected. Loaded {len(gates)} gates.")
    print("[GateTest] Drive manually and watch the gate counter below.")
    print("[GateTest] Press Ctrl+C to stop.\n")

    prev_pos = None
    current_gate_idx = 0
    gates_crossed = 0

    try:
        while True:
            state = bridge.recv_state()
            bridge.send_action("MANUAL")

            cur_pos = state.position
            speed = state.display_speed
            yaw_deg = math.degrees(state.yaw)

            if prev_pos is None:
                prev_pos = cur_pos
                print(
                    f"[GateTest] start  pos=({cur_pos[0]:.1f}, {cur_pos[1]:.1f}, {cur_pos[2]:.1f}) "
                    f"speed={speed:.1f} km/h yaw={yaw_deg:.1f}°"
                )
                continue

            nearest_gate_idx = min(
                range(current_gate_idx, len(gates)),
                key=lambda idx: _dist_to_gate_center(cur_pos, gates[idx]),
            ) if current_gate_idx < len(gates) else None

            crossed_any = False
            # Catch up through multiple gates if the telemetry jump spans more than one.
            while current_gate_idx < len(gates):
                gate = gates[current_gate_idx]
                if _check_gate_crossing(prev_pos, cur_pos, gate, cfg):
                    gates_crossed += 1
                    crossed_any = True
                    print(
                        f"[GateTest] GATE {gate['gate_id']} crossed "
                        f"({gates_crossed}/{len(gates)})  "
                        f"pos=({cur_pos[0]:.1f}, {cur_pos[1]:.1f}, {cur_pos[2]:.1f}) "
                        f"speed={speed:.1f} km/h"
                    )
                    current_gate_idx += 1
                    continue
                break

            if not crossed_any and nearest_gate_idx is not None:
                nearest_gate = gates[nearest_gate_idx]
                nearest_dist = _dist_to_gate_center(cur_pos, nearest_gate)
                if nearest_gate_idx == current_gate_idx:
                    print(
                        f"[GateTest] near gate {nearest_gate['gate_id']} "
                        f"dist={nearest_dist:.1f}m expected={current_gate_idx} "
                        f"pos=({cur_pos[0]:.1f}, {cur_pos[1]:.1f}, {cur_pos[2]:.1f})"
                    )

            prev_pos = cur_pos

    except KeyboardInterrupt:
        print("\n[GateTest] Stopped by user.")
    finally:
        try:
            bridge.send_action("MANUAL")
        except Exception:
            pass
        bridge.close()


def main():
    parser = argparse.ArgumentParser(description="Manual Trackmania gate monitor")
    parser.add_argument("--port", type=int, default=9000, help="telemetry.as socket port (default: 9000)")
    args = parser.parse_args()
    monitor(args.port)


if __name__ == "__main__":
    main()