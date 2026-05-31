"""
Thin Gymnasium environment for Trackmania Nations Forever — GATE-BASED.

Loads data/gates.json (produced by generate_gates.py) and rewards the agent
for crossing gates in sequential order.

Delegates ALL reward/termination logic to rewards.compute().
Owns: connection, observation, action mapping, gate geometry, reset, logging.
Does NOT own: reward math, termination decisions.
"""

from __future__ import annotations

import math
import json
import os
import time
from collections import deque
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from telemetry_bridge import TelemetryBridge
from rewards import (
    RewardConfig, RewardContext, EpisodeState, RewardBreakdown,
    compute, segments_intersect_2d,
)
from state_normaliser import StateNormalizer


# ---------------------------------------------------------------------------
# Episode stats
# ---------------------------------------------------------------------------

class _EpisodeRecord:
    """Lightweight snapshot of a finished episode for aggregate stats."""
    __slots__ = ("reason", "reward", "steps", "gates_crossed", "max_speed",
                 "game_time_ms", "total_gates")

    def __init__(self, reason, reward, steps, gates_crossed, max_speed,
                 game_time_ms, total_gates):
        self.reason = reason
        self.reward = reward
        self.steps = steps
        self.gates_crossed = gates_crossed
        self.max_speed = max_speed
        self.game_time_ms = game_time_ms
        self.total_gates = total_gates


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class TrackmaniaEnv(gym.Env):
    """
    Gymnasium env for TMNF via telemetry.as TCP bridge — gate-based rewards.

    Observation space (14 floats, pre-scaled ≈[-1,1] before VecNormalize):
        [speed_norm, yaw_norm, pitch_norm, roll_norm,
         vel_local_x/100, vel_local_y/100, vel_local_z/100,
         gate_dx/100, gate_dz/100,        -- vector to next gate center (car-local)
         gate_dist/200,                    -- distance to next gate center
         gate_progress,                    -- fraction of gates completed
         prev_steer, prev_gas, prev_brake]

    Action space — continuous Box(3,) in [-1, 1]:
        [steer, gas, brake]
    """

    metadata = {"render_modes": ["human"]}

    TRAINING_SPEED = 1.0
    TRACK_SURFACE_Y = 26.0

    OBS_DIM = 14

    # How often (in episodes) to print aggregate stats
    STATS_EVERY = 10

    def __init__(self, port: int = 9000, ticks_per_step: int = 25,
                 reward_cfg: Optional[RewardConfig] = None):
        super().__init__()

        self.port = port
        self.ticks_per_step = ticks_per_step
        self.reward_cfg = reward_cfg or RewardConfig()

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.OBS_DIM,), dtype=np.float32,
        )

        # -- Gates --
        self.gates = []
        self._gate_centers = None    # np array shape (N, 3)
        self._gate_lefts = None      # np array shape (N, 3)
        self._gate_rights = None     # np array shape (N, 3)
        self._num_gates = 0

        gates_path = os.path.join(
            os.path.dirname(__file__), "data", "gates.json",
        )
        if os.path.exists(gates_path):
            with open(gates_path, "r") as f:
                gate_data = json.load(f)
            gate_list = gate_data.get("gates", gate_data) if isinstance(gate_data, dict) else gate_data
            self.gates = gate_list
            self._num_gates = len(gate_list)

            if self._num_gates > 0:
                self._gate_centers = np.array(
                    [g["center"] for g in gate_list], dtype=np.float64
                )
                self._gate_lefts = np.array(
                    [g["left_post"] for g in gate_list], dtype=np.float64
                )
                self._gate_rights = np.array(
                    [g["right_post"] for g in gate_list], dtype=np.float64
                )

            print(
                f"[Port {self.port}] Loaded {self._num_gates} gates "
                f"(track arc ≈ {gate_data.get('total_arc_m', 0):.0f} m)"
            )
        else:
            print(
                f"[Port {self.port}] WARNING: data/gates.json not found. "
                "Run record_waypoints.py then generate_gates.py first."
            )

        # -- Walls --
        self._wall_points_xz = np.empty((0, 2), dtype=np.float64)
        self._load_wall_points()

        # -- Episode state --
        self._ep_state = EpisodeState()
        self._step_count = 0
        self._prev_action = np.zeros(3, dtype=np.float32)
        self._prev_position = np.zeros(3, dtype=np.float64)
        self._last_race_time_ms = 0

        # -- Connection --
        self.bridge = TelemetryBridge(port=self.port)
        self.connected = False
        self.current_state = None
        self.previous_speed = 0.0
        self.normalizer = StateNormalizer()

        # -- Logging --
        self._episode_count = 0
        self._finish_count_total = 0
        self._recent_episodes: deque[_EpisodeRecord] = deque(maxlen=50)
        self._wall_start = 0.0
        self._episode_reward_total = 0.0
        self._episode_breakdown_total = RewardBreakdown()
        self._no_move_giveup_pulsed = False

    # ==================================================================
    # GATE GEOMETRY
    # ==================================================================

    def _check_gate_crossing(self, prev_pos, cur_pos, gate_idx):
        """
        Count a gate as passed when the car path segment intersects the
        gate bar segment (left_post -> right_post) in XZ.

        This is angle-agnostic by construction: any crossing of the bar counts.
        """
        if gate_idx >= self._num_gates:
            return False

        gate = self.gates[gate_idx]
        center = gate["center"]
        if abs(float(cur_pos[1]) - float(center[1])) > self.reward_cfg.gate_y_tolerance:
            return False

        left = gate["left_post"]
        right = gate["right_post"]

        p1 = (float(prev_pos[0]), float(prev_pos[2]))
        p2 = (float(cur_pos[0]), float(cur_pos[2]))
        q1 = (float(left[0]), float(left[2]))
        q2 = (float(right[0]), float(right[2]))
        return segments_intersect_2d(p1, p2, q1, q2)

    def _find_crossed_gate_idx(self, prev_pos, cur_pos, expected_gate_idx):
        """Find the furthest crossed gate from expected index onward."""
        if self._num_gates <= 0 or expected_gate_idx >= self._num_gates:
            return None

        furthest_idx = None
        for idx in range(expected_gate_idx, self._num_gates):
            if self._check_gate_crossing(prev_pos, cur_pos, idx):
                furthest_idx = idx

        return furthest_idx

    def _dist_to_gate_center(self, pos, gate_idx):
        """Euclidean distance from pos to gate center (3D)."""
        if gate_idx >= self._num_gates:
            return 0.0
        c = self._gate_centers[gate_idx]
        return float(np.linalg.norm(np.array(pos[:3], dtype=np.float64) - c))

    def _load_wall_points(self):
        """Load recorded wall samples for proximity-based shaping."""
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        point_sets = []

        for name in ("left_wall_points.npy", "right_wall_points.npy"):
            path = os.path.join(data_dir, name)
            if not os.path.exists(path):
                continue
            arr = np.asarray(np.load(path), dtype=np.float64)
            if arr.ndim != 2:
                continue
            if arr.shape[1] >= 3:
                pts = arr[:, [0, 2]]
            elif arr.shape[1] >= 2:
                pts = arr[:, :2]
            else:
                continue
            point_sets.append(pts)

        if not point_sets:
            wall_json = os.path.join(data_dir, "wall_points.json")
            if os.path.exists(wall_json):
                with open(wall_json, "r") as f:
                    wall_data = json.load(f)

                if isinstance(wall_data, dict):
                    for key in ("left_wall_points_xz", "right_wall_points_xz", "points_xz"):
                        pts = wall_data.get(key)
                        if not pts:
                            continue
                        arr = np.asarray(pts, dtype=np.float64)
                        if arr.ndim == 2 and arr.shape[1] >= 2:
                            point_sets.append(arr[:, :2])

        if point_sets:
            self._wall_points_xz = np.concatenate(point_sets, axis=0)
            print(f"[Port {self.port}] Loaded {len(self._wall_points_xz)} wall points")
        else:
            print(f"[Port {self.port}] WARNING: no wall points found for proximity penalty")

    def _dist_to_nearest_wall(self, pos):
        """Nearest XZ distance to recorded wall samples."""
        if self._wall_points_xz.size == 0:
            return 999.0

        p = np.array([float(pos[0]), float(pos[2])], dtype=np.float64)
        deltas = self._wall_points_xz - p
        dist_sq = np.einsum("ij,ij->i", deltas, deltas)
        return float(np.sqrt(np.min(dist_sq)))

    # ==================================================================
    # OBSERVATION
    # ==================================================================

    def _get_observation(self, state):
        """
        14-float observation:
          [speed/300, yaw/π, pitch/π, roll/π,
           vel_local_x/100, vel_local_y/100, vel_local_z/100,
           gate_dx/100, gate_dz/100,
           gate_dist/200,
           gate_progress,
           prev_steer, prev_gas, prev_brake]
        """
        speed = state.display_speed
        pos = state.position
        yaw, pitch, roll = state.yaw_pitch_roll
        vel = state.velocity

        # Car-local velocity
        cos_y = math.cos(-yaw)
        sin_y = math.sin(-yaw)
        vel_local_x = (vel[0] * cos_y - vel[2] * sin_y) / 100.0
        vel_local_y = vel[1] / 100.0
        vel_local_z = (vel[0] * sin_y + vel[2] * cos_y) / 100.0

        # Vector to next gate center in car-local frame
        gate_dx_local = 0.0
        gate_dz_local = 0.0
        gate_dist = 0.0
        gate_progress = 0.0

        gate_idx = self._ep_state.current_gate_idx
        if self._num_gates > 0 and gate_idx < self._num_gates:
            gc = self._gate_centers[gate_idx]
            dx = gc[0] - pos[0]
            dz = gc[2] - pos[2]
            gate_dx_local = (dx * cos_y - dz * sin_y) / 100.0
            gate_dz_local = (dx * sin_y + dz * cos_y) / 100.0
            gate_dist = math.sqrt(dx * dx + dz * dz) / 200.0
            gate_progress = gate_idx / max(self._num_gates, 1)

        raw_obs = np.array([
            float(speed) / 300.0,
            float(yaw) / math.pi,
            float(pitch) / math.pi,
            float(roll) / math.pi,
            float(vel_local_x),
            float(vel_local_y),
            float(vel_local_z),
            float(gate_dx_local),
            float(gate_dz_local),
            float(gate_dist),
            float(gate_progress),
            self._prev_action[0],
            self._prev_action[1],
            self._prev_action[2],
        ], dtype=np.float32)

        raw_obs = np.nan_to_num(raw_obs, nan=0.0, posinf=5.0, neginf=-5.0)
        return self.normalizer.normalize(raw_obs)

    # ==================================================================
    # ACTION
    # ==================================================================

    def _action_to_token(self, action) -> str:
        """Map continuous policy action to telemetry.as discrete action tokens."""
        steer = float(np.clip(action[0], -1.0, 1.0))
        # Slight accel bias prevents early-policy collapse into coasting.
        gas = float(action[1]) > -0.25
        # Require deliberate brake intent so random noise doesn't spam coast.
        brake = float(action[2]) > 0.60

        if brake:
            return "COAST"
        if not gas:
            return "COAST"
        if steer <= -0.25:
            return "LEFT"
        if steer >= 0.25:
            return "RIGHT"
        return "ACCEL"

    # ==================================================================
    # CONNECTION
    # ==================================================================

    def connect(self):
        if not self.connected:
            print(f"[Port {self.port}] Waiting for telemetry.as connection ...")
            self.bridge.connect()
            self.connected = True
            self.bridge.send_action("MANUAL")
            print(f"[Port {self.port}] Connected to telemetry.as client.")

    # ==================================================================
    # STEP
    # ==================================================================

    def step(self, action):
        if not self.connected:
            self.connect()

        action_token = self._action_to_token(action)
        try:
            self.bridge.send_action(action_token)
            state = self.bridge.recv_state()
        except (ConnectionResetError, BrokenPipeError, TimeoutError, ConnectionError, OSError) as e:
            print(f"[Port {self.port}] Bridge disconnected ({e}); stopping training so you can restart cleanly.")
            try:
                self.bridge.close()
            except Exception:
                pass
            self.connected = False
            raise ConnectionError(f"Bridge disconnected: {e}")

        race_time = state.sample_idx * 100

        cur_pos = np.array(state.position, dtype=np.float64)
        prev_pos = self._prev_position.copy()

        # -- Gate crossing detection --
        gate_idx = self._ep_state.current_gate_idx
        gate_idx_before = gate_idx
        gate_crossed = False
        is_final_gate = False
        crossed_gate_idx = None
        gate_resynced_from = None
        gate_advanced_by = 0

        if self._num_gates > 0 and gate_idx < self._num_gates:
            crossed_gate_idx = self._find_crossed_gate_idx(prev_pos, cur_pos, gate_idx)
            gate_crossed = crossed_gate_idx is not None
            if gate_crossed:
                if crossed_gate_idx != gate_idx:
                    gate_resynced_from = gate_idx
                    # If multiple bars were crossed in one step, pre-credit all
                    # skipped gates so episode progress stays aligned.
                    gate_advanced_by = crossed_gate_idx - gate_idx
                    if gate_advanced_by > 0:
                        self._ep_state.gates_crossed += gate_advanced_by
                    self._ep_state.current_gate_idx = crossed_gate_idx
                    gate_idx = crossed_gate_idx
                is_final_gate = (crossed_gate_idx == self._num_gates - 1)

        dist_to_gate = self._dist_to_gate_center(
            cur_pos,
            min(gate_idx, self._num_gates - 1) if self._num_gates > 0 else 0,
        )
        dist_to_wall = self._dist_to_nearest_wall(cur_pos)

        ctx = RewardContext(
            position=cur_pos,
            prev_position=prev_pos,
            speed_kmh=state.display_speed,
            prev_speed_kmh=self.previous_speed,
            action=np.array(action, dtype=np.float32),
            prev_action=self._prev_action.copy(),
            step_idx=self._step_count,
            race_time_ms=race_time,
            dist_to_gate_m=dist_to_gate,
            dist_to_wall_m=dist_to_wall,
            gate_crossed=gate_crossed,
            is_final_gate=is_final_gate,
            num_gates=self._num_gates,
        )

        result = compute(ctx, self._ep_state, self.reward_cfg)
        self._ep_state = result.episode_state

        forced_no_move_reset = False
        if (
            not result.terminated
            and not result.truncated
            and not self._ep_state.has_moved
            and self._ep_state.gates_crossed == 0
            and self._step_count >= self.reward_cfg.stuck_steps
            and not self._no_move_giveup_pulsed
        ):
            # Pulse RESET once for no-move stalls, then end episode so reset runs.
            try:
                self.bridge.send_action("RESET")
            except Exception as e:
                print(f"[Port {self.port}] RESET pulse failed ({e})")
            self._no_move_giveup_pulsed = True
            forced_no_move_reset = True

        final_reward = float(result.reward)
        final_terminated = bool(result.terminated)
        final_truncated = bool(result.truncated)
        final_reason = result.reason

        if forced_no_move_reset:
            final_reward += float(self.reward_cfg.stuck_penalty)
            final_terminated = True
            final_reason = "stuck_no_move"

        self._episode_reward_total += float(result.reward)
        self._episode_breakdown_total.gate_cross += float(result.episode_state.breakdown.gate_cross)
        self._episode_breakdown_total.approach += float(result.episode_state.breakdown.approach)
        self._episode_breakdown_total.speed += float(result.episode_state.breakdown.speed)
        self._episode_breakdown_total.braking += float(result.episode_state.breakdown.braking)
        self._episode_breakdown_total.time_cost += float(result.episode_state.breakdown.time_cost)
        self._episode_breakdown_total.finish += float(result.episode_state.breakdown.finish)
        self._episode_breakdown_total.penalties += float(result.episode_state.breakdown.penalties)
        self._episode_breakdown_total.fell_off += float(result.episode_state.breakdown.fell_off)
        self._episode_breakdown_total.wall_hit += float(result.episode_state.breakdown.wall_hit)
        self._episode_breakdown_total.wall_proximity += float(result.episode_state.breakdown.wall_proximity)
        self._episode_breakdown_total.crash += float(result.episode_state.breakdown.crash)
        self._episode_breakdown_total.stuck += float(result.episode_state.breakdown.stuck)
        self._episode_breakdown_total.wrong_way += float(result.episode_state.breakdown.wrong_way)
        if forced_no_move_reset:
            self._episode_reward_total += float(self.reward_cfg.stuck_penalty)
            self._episode_breakdown_total.stuck += float(self.reward_cfg.stuck_penalty)
            self._episode_breakdown_total.penalties += float(self.reward_cfg.stuck_penalty)
            self._episode_breakdown_total.total += float(self.reward_cfg.stuck_penalty)

        if final_reason == "finish":
            self._finish_count_total += 1
        self.previous_speed = state.display_speed
        self._prev_action = np.array(action, dtype=np.float32)
        self._prev_position = cur_pos.copy()
        self._step_count += 1
        self._last_race_time_ms = race_time

        gate_pct = 0.0
        if self._num_gates > 0:
            gate_pct = self._ep_state.gates_crossed / self._num_gates * 100.0

        info = {
            "termination_reason": final_reason,
            "gates_crossed": self._ep_state.gates_crossed,
            "gates_total": self._num_gates,
            "gate_progress_pct": gate_pct,
            "current_gate_idx": self._ep_state.current_gate_idx,
            "dist_to_gate_m": dist_to_gate,
            "dist_to_wall_m": dist_to_wall,
            "crossed_gate_idx": crossed_gate_idx,
            "gate_resynced_from": gate_resynced_from,
            "gate_advanced_by": gate_advanced_by,
            "gate_idx_before": gate_idx_before,
            "raw_reward": result.reward,
            "speed_kmh": state.display_speed,
            "step_idx": self._step_count,
            "max_speed_kmh": self._ep_state.max_speed,
            "finish_count_total": self._finish_count_total,
            "forced_no_move_reset": forced_no_move_reset,
        }

        if final_terminated or final_truncated:
            self._log_episode_end(final_reason, gate_pct)

        obs = self._get_observation(state)
        self.current_state = state
        return obs, final_reward, final_terminated, final_truncated, info

    # ==================================================================
    # RESET
    # ==================================================================

    def reset(self, seed=None, options=None):
        """Always restart from the start line."""
        super().reset(seed=seed)
        if not self.connected:
            self.connect()

        # Fresh episode state
        self._ep_state = EpisodeState()
        # Explicitly clear gate progress/reward accounting each reset.
        self._ep_state.gates_crossed = 0
        self._ep_state.current_gate_idx = 0
        self._step_count = 0
        self._prev_action = np.zeros(3, dtype=np.float32)
        self._prev_position = np.zeros(3, dtype=np.float64)
        self._last_race_time_ms = 0
        self._wall_start = time.monotonic()
        self._episode_count += 1
        self._episode_reward_total = 0.0
        self._episode_breakdown_total = RewardBreakdown()
        self._no_move_giveup_pulsed = False

        # Ask plugin to respawn, then consume a few telemetry frames.
        self.bridge.send_action("RESET")
        state = None
        for _ in range(4):
            state = self.bridge.recv_state()
        self.bridge.send_action("MANUAL")

        # Initialise previous position from spawn
        if state is not None:
            self._prev_position = np.array(state.position, dtype=np.float64)
            self.previous_speed = 0.0

            # In practice reset often places the car on/just past gate 0;
            # start at gate 1 to avoid deadlocking progression on gate 0.
            if self._num_gates > 1:
                self._ep_state.current_gate_idx = 1

        obs = self._get_observation(state)
        self.current_state = state
        self._log_episode_start()
        return obs, {}

    # ==================================================================
    # LOGGING
    # ==================================================================

    def _log_episode_start(self):
        print(
            f"[Port {self.port}] Episode #{self._episode_count} started "
            f"(gates={self._num_gates}, start_gate_idx={self._ep_state.current_gate_idx}, "
            f"gates_crossed={self._ep_state.gates_crossed}, reward_total={self._episode_reward_total:+.2f})"
        )

    def _log_episode_end(self, reason: str, gate_pct: float):
        bd = self._episode_breakdown_total
        wall_sec = time.monotonic() - self._wall_start
        game_sec = self._last_race_time_ms / 1000.0

        print(f"\n{'=' * 60}")
        print(f"  [Port {self.port}] Episode #{self._episode_count}  ENDED")
        print(f"{'=' * 60}")
        print(f"  Reason       : {reason}")
        print(f"  Total finishes: {self._finish_count_total}")
        print(f"  Steps        : {self._step_count}")
        print(f"  Game time    : {game_sec:.1f}s   Wall time: {wall_sec:.1f}s")
        print(f"  Gates crossed: {self._ep_state.gates_crossed} / "
              f"{self._num_gates}  ({gate_pct:.1f}%)")
        print(f"  Max speed    : {self._ep_state.max_speed:.1f} km/h")
        print(f"  Total reward : {self._episode_reward_total:+.2f}")
        print(f"    gate_cross : {bd.gate_cross:+.3f}")
        print(f"    approach   : {bd.approach:+.3f}")
        print(f"    speed      : {bd.speed:+.3f}")
        print(f"    braking    : {bd.braking:+.3f}")
        print(f"    time_cost  : {bd.time_cost:+.3f}")
        print(f"    finish     : {bd.finish:+.3f}")
        print(f"    penalties  : {bd.penalties:+.3f}  "
              f"(fell={bd.fell_off:.1f} wall={bd.wall_hit:.1f} wall_prox={bd.wall_proximity:.1f} "
              f"crash={bd.crash:.1f} stuck={bd.stuck:.1f} "
              f"wrong_way={bd.wrong_way:.1f})")
        print(f"  Stuck ctr    : {self._ep_state.consecutive_stuck} / "
              f"{self.reward_cfg.stuck_steps}  "
              f"has_moved={self._ep_state.has_moved}")
        print(f"  Wrong-way ctr: {self._ep_state.consecutive_wrong_way} / "
              f"{self.reward_cfg.wrong_way_steps}")
        print(f"{'=' * 60}\n")

        self._recent_episodes.append(_EpisodeRecord(
            reason=reason,
            reward=self._episode_reward_total,
            steps=self._step_count,
            gates_crossed=self._ep_state.gates_crossed,
            max_speed=self._ep_state.max_speed,
            game_time_ms=self._last_race_time_ms,
            total_gates=self._num_gates,
        ))

        if self._episode_count % self.STATS_EVERY == 0:
            self._log_aggregate_stats()

    def _log_aggregate_stats(self):
        eps = list(self._recent_episodes)
        if not eps:
            return

        n = min(len(eps), self.STATS_EVERY)
        recent = eps[-n:]

        avg_rew = sum(e.reward for e in recent) / n
        avg_steps = sum(e.steps for e in recent) / n
        avg_gates = sum(e.gates_crossed for e in recent) / n
        avg_speed = sum(e.max_speed for e in recent) / n

        reasons: dict[str, int] = {}
        for e in recent:
            reasons[e.reason] = reasons.get(e.reason, 0) + 1
        reason_str = "  ".join(f"{k}({v})" for k, v in sorted(reasons.items()))

        print(f"{'#' * 60}")
        print(f"  [Port {self.port}] STATS — last {n} episodes "
              f"(total: {self._episode_count})")
        print(f"{'#' * 60}")
        print(f"  Avg reward    : {avg_rew:+.2f}")
        print(f"  Avg steps     : {avg_steps:.0f}")
        print(f"  Avg gates     : {avg_gates:.1f} / {self._num_gates}")
        print(f"  Avg max speed : {avg_speed:.1f} km/h")
        print(f"  Reasons       : {reason_str}")
        print(f"{'#' * 60}\n")

    # ==================================================================
    # CLOSE
    # ==================================================================

    def close(self):
        if self.connected:
            self.bridge.close()
            self.connected = False


GateBridgeEnv = TrackmaniaEnv
