"""
Thin Gymnasium environment for Trackmania Nations Forever — GATE-BASED.

Loads data/gates.json (produced by generate_gates.py) and rewards the agent
for crossing gates in sequential order.

Delegates ALL reward/termination logic to rewards.compute().
Owns: connection, observation, action mapping, gate geometry, reset, logging.
Does NOT own: reward math, termination decisions.
"""

from __future__ import annotations

import random
import math
import json
import os
import sys
import time
from collections import deque
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .tminterface2 import TMInterface, MessageType
from .rewards import (
    RewardConfig, RewardContext, EpisodeState, RewardBreakdown,
    compute, segments_intersect_2d,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_processing.features import StateNormalizer


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
    Gymnasium env for TMNF via TMInterface sockets — gate-based rewards.

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

    TRAINING_SPEED = 5.0
    TRACK_SURFACE_Y = 26.0

    OBS_DIM = 14

    # How often (in episodes) to print aggregate stats
    STATS_EVERY = 10

    def __init__(self, port: int = 8483, ticks_per_step: int = 25,
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
            os.path.dirname(os.path.dirname(__file__)), "data", "gates.json",
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

        # -- Episode state --
        self._ep_state = EpisodeState()
        self._step_count = 0
        self._prev_action = np.zeros(3, dtype=np.float32)
        self._prev_position = np.zeros(3, dtype=np.float64)
        self._last_race_time_ms = 0
        self._cp_current = 0
        self._cp_target = 0

        # -- Connection --
        self.iface = TMInterface(self.port)
        self.connected = False
        self.current_state = None
        self.previous_speed = 0.0
        self.normalizer = StateNormalizer()

        # -- Logging --
        self._episode_count = 0
        self._recent_episodes: deque[_EpisodeRecord] = deque(maxlen=50)
        self._wall_start = 0.0

    # ==================================================================
    # GATE GEOMETRY
    # ==================================================================

    def _check_gate_crossing(self, prev_pos, cur_pos, gate_idx):
        """
        Check if the car's movement segment crosses the target gate.
        Uses 2D intersection in the XZ plane, with Y tolerance check.
        """
        if gate_idx >= self._num_gates:
            return False

        gate = self.gates[gate_idx]
        left = gate["left_post"]
        right = gate["right_post"]
        center = gate["center"]

        # Y tolerance: car must be roughly at gate height
        y_tol = self.reward_cfg.gate_y_tolerance
        if abs(cur_pos[1] - center[1]) > y_tol:
            return False

        # 2D segment intersection in XZ plane
        p1 = (prev_pos[0], prev_pos[2])
        p2 = (cur_pos[0], cur_pos[2])
        q1 = (left[0], left[2])
        q2 = (right[0], right[2])

        return segments_intersect_2d(p1, p2, q1, q2)

    def _dist_to_gate_center(self, pos, gate_idx):
        """Euclidean distance from pos to gate center (3D)."""
        if gate_idx >= self._num_gates:
            return 0.0
        c = self._gate_centers[gate_idx]
        return float(np.linalg.norm(np.array(pos[:3], dtype=np.float64) - c))

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

    def _apply_action(self, action):
        """Map continuous [-1,1]^3 to TMInterface inputs."""
        steer = int(np.clip(float(action[0]), -1.0, 1.0) * 65536)
        accelerate = bool(float(action[1]) > 0.0)
        brake = bool(float(action[2]) > 0.0)
        self.iface.set_input_state(accelerate=accelerate, brake=brake, steer=steer)

    # ==================================================================
    # CONNECTION
    # ==================================================================

    def connect(self):
        if not self.connected:
            print(f"[Port {self.port}] Connecting to TMInterface ...")
            self.iface.register(timeout=None)
            self.connected = True

            while True:
                msgtype = self.iface._read_int32()
                if msgtype == int(MessageType.SC_ON_CONNECT_SYNC):
                    self.iface.set_on_step_period(self.ticks_per_step)
                    self.iface.set_speed(self.TRAINING_SPEED)
                    self.iface._respond_to_call(msgtype)
                    break
                else:
                    self.iface._respond_to_call(msgtype)

            print(f"[Port {self.port}] Connected — simulation speed {self.TRAINING_SPEED}x.")

    # ==================================================================
    # STEP
    # ==================================================================

    def step(self, action):
        if not self.connected:
            self.connect()

        self._apply_action(action)

        # Accumulate checkpoint events
        cp_hit_this_step = False
        is_finish = False

        while True:
            msgtype = self.iface._read_int32()

            if msgtype == int(MessageType.SC_RUN_STEP_SYNC):
                race_time = self.iface._read_int32()
                state = self.iface.get_simulation_state()

                cur_pos = np.array(state.position, dtype=np.float64)
                prev_pos = self._prev_position.copy()

                # -- Gate crossing detection --
                gate_idx = self._ep_state.current_gate_idx
                gate_crossed = False
                is_final_gate = False

                if self._num_gates > 0 and gate_idx < self._num_gates:
                    gate_crossed = self._check_gate_crossing(
                        prev_pos, cur_pos, gate_idx
                    )
                    is_final_gate = (gate_idx == self._num_gates - 1)

                # Distance to current target gate
                dist_to_gate = self._dist_to_gate_center(
                    cur_pos,
                    min(gate_idx, self._num_gates - 1) if self._num_gates > 0 else 0
                )

                # -- Build reward context --
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
                    gate_crossed=gate_crossed,
                    is_final_gate=is_final_gate,
                    num_gates=self._num_gates,
                )

                # -- Compute reward --
                result = compute(ctx, self._ep_state, self.reward_cfg)
                self._ep_state = result.episode_state
                self.previous_speed = state.display_speed
                self._prev_action = np.array(action, dtype=np.float32)
                self._prev_position = cur_pos.copy()
                self._step_count += 1
                self._last_race_time_ms = race_time

                # -- Build info dict --
                gate_pct = 0.0
                if self._num_gates > 0:
                    gate_pct = self._ep_state.gates_crossed / self._num_gates * 100.0

                info = {
                    "termination_reason": result.reason,
                    "gates_crossed": self._ep_state.gates_crossed,
                    "gates_total": self._num_gates,
                    "gate_progress_pct": gate_pct,
                    "current_gate_idx": self._ep_state.current_gate_idx,
                    "dist_to_gate_m": dist_to_gate,
                    "raw_reward": result.reward,
                    "speed_kmh": state.display_speed,
                    "step_idx": self._step_count,
                    "max_speed_kmh": self._ep_state.max_speed,
                }

                # -- Log episode end --
                if result.terminated or result.truncated:
                    self._log_episode_end(result.reason, gate_pct)

                obs = self._get_observation(state)
                self.current_state = state
                self.iface._respond_to_call(msgtype)
                return obs, result.reward, result.terminated, result.truncated, info

            elif msgtype == int(MessageType.SC_CHECKPOINT_COUNT_CHANGED_SYNC):
                self._cp_current = self.iface._read_int32()
                self._cp_target = self.iface._read_int32()
                cp_hit_this_step = True
                if self._cp_current == self._cp_target:
                    is_finish = True
                    print(f"[Port {self.port}] *** CHECKPOINT FINISH ***")
                self.iface._respond_to_call(msgtype)

            else:
                self.iface._respond_to_call(msgtype)

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
        self._step_count = 0
        self._prev_action = np.zeros(3, dtype=np.float32)
        self._prev_position = np.zeros(3, dtype=np.float64)
        self._last_race_time_ms = 0
        self._cp_current = 0
        self._cp_target = 0
        self._wall_start = time.monotonic()
        self._episode_count += 1

        # Give up → restart race from start line
        self.iface.give_up()

        first_frame = True
        prev_time = -1
        state = None
        while True:
            msgtype = self.iface._read_int32()
            if msgtype == int(MessageType.SC_RUN_STEP_SYNC):
                _time = self.iface._read_int32()
                state = self.iface.get_simulation_state()
                self.iface._respond_to_call(msgtype)

                if first_frame:
                    prev_time = _time
                    first_frame = False
                    continue

                if (_time == 0) or (_time > 0 and _time < prev_time - 100):
                    break
                prev_time = _time
            else:
                self.iface._respond_to_call(msgtype)

        # Initialise previous position from spawn
        if state is not None:
            self._prev_position = np.array(state.position, dtype=np.float64)
            self.previous_speed = 0.0

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
            f"(gates={self._num_gates})"
        )

    def _log_episode_end(self, reason: str, gate_pct: float):
        bd = self._ep_state.breakdown
        wall_sec = time.monotonic() - self._wall_start
        game_sec = self._last_race_time_ms / 1000.0

        print(f"\n{'=' * 60}")
        print(f"  [Port {self.port}] Episode #{self._episode_count}  ENDED")
        print(f"{'=' * 60}")
        print(f"  Reason       : {reason}")
        print(f"  Steps        : {self._step_count}")
        print(f"  Game time    : {game_sec:.1f}s   Wall time: {wall_sec:.1f}s")
        print(f"  Gates crossed: {self._ep_state.gates_crossed} / "
              f"{self._num_gates}  ({gate_pct:.1f}%)")
        print(f"  Max speed    : {self._ep_state.max_speed:.1f} km/h")
        print(f"  Total reward : {bd.total:+.2f}")
        print(f"    gate_cross : {bd.gate_cross:+.3f}")
        print(f"    approach   : {bd.approach:+.3f}")
        print(f"    speed      : {bd.speed:+.3f}")
        print(f"    time_cost  : {bd.time_cost:+.3f}")
        print(f"    finish     : {bd.finish:+.3f}")
        print(f"    penalties  : {bd.penalties:+.3f}  "
              f"(fell={bd.fell_off:.1f} crash={bd.crash:.1f} stuck={bd.stuck:.1f} "
              f"wrong_way={bd.wrong_way:.1f})")
        print(f"  Stuck ctr    : {self._ep_state.consecutive_stuck} / "
              f"{self.reward_cfg.stuck_steps}  "
              f"has_moved={self._ep_state.has_moved}")
        print(f"  Wrong-way ctr: {self._ep_state.consecutive_wrong_way} / "
              f"{self.reward_cfg.wrong_way_steps}")
        print(f"{'=' * 60}\n")

        self._recent_episodes.append(_EpisodeRecord(
            reason=reason,
            reward=bd.total,
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
            self.iface.close()
            self.connected = False
