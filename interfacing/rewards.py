"""
Reward and termination logic — GATE-BASED system.

PURE MODULE — no TM dependencies, no sockets, no GPU.
Unit-testable with fake RewardContext objects.

Design (from implementation_plan.md):
  - Primary reward: crossing the next gate in sequence
  - Shaping: closing distance to next gate center
  - Terminators: stuck, fell-off, wrong-way, timeout
  - Track width: ~32 m (standard TMNF Stadium)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np
import math


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RewardConfig:
    """All tuneable constants in one place."""

    # --- Gate crossing ---
    gate_reward: float = 20.0          # reward per gate crossed
    finish_bonus: float = 200.0        # extra reward for crossing final gate

    # --- Dense shaping ---
    approach_weight: float = 0.05      # reward for closing distance to next gate
    speed_weight: float = 0.0          # speed reward (disabled by default)
    speed_max_kmh: float = 300.0
    time_penalty: float = 0.05         # per-step time penalty

    # --- Terminators ---
    fell_off_y: float = 20.0           # Y below this = fell off
    fell_off_penalty: float = 5.0
    crash_speed_drop_kmh: float = 40.0  # sudden drop = wall hit
    crash_penalty: float = 10.0
    stuck_speed_thresh: float = 15.0   # km/h
    stuck_steps: int = 30              # consecutive steps below threshold
    stuck_penalty: float = 5.0
    wrong_way_dist_m: float = 80.0     # if dist to gate increases past this, terminate
    wrong_way_steps: int = 30          # consecutive steps getting further away
    wrong_way_penalty: float = 5.0
    timeout_steps: int = 3000

    # --- Grace ---
    move_threshold_kmh: float = 20.0   # must exceed before stuck can fire
    startup_stuck_steps: int = 200     # terminate if never moves at all

    # --- Gate intersection ---
    gate_y_tolerance: float = 10.0     # vertical tolerance for gate crossing (metres)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RewardBreakdown:
    """Cumulative reward by component — printed at episode end."""
    gate_cross: float = 0.0
    approach: float = 0.0
    speed: float = 0.0
    time_cost: float = 0.0
    finish: float = 0.0
    fell_off: float = 0.0
    crash: float = 0.0
    stuck: float = 0.0
    wrong_way: float = 0.0

    @property
    def total(self) -> float:
        return (self.gate_cross + self.approach + self.speed + self.time_cost
                + self.finish
                + self.fell_off + self.crash + self.stuck + self.wrong_way)

    @property
    def penalties(self) -> float:
        return self.fell_off + self.crash + self.stuck + self.wrong_way


@dataclass
class EpisodeState:
    """Mutable carry-over state between steps. Owned by env, mutated by compute()."""
    current_gate_idx: int = 0
    prev_dist_to_gate: float = float("inf")
    consecutive_stuck: int = 0
    consecutive_wrong_way: int = 0
    has_moved: bool = False
    max_speed: float = 0.0
    gates_crossed: int = 0
    breakdown: RewardBreakdown = field(default_factory=RewardBreakdown)


@dataclass
class RewardContext:
    """Snapshot of one step — built by env.py, consumed by compute()."""
    position: np.ndarray          # [x, y, z]
    prev_position: np.ndarray     # [x, y, z] from previous step
    speed_kmh: float
    prev_speed_kmh: float
    action: np.ndarray
    prev_action: np.ndarray
    step_idx: int
    race_time_ms: int
    # Gate info (set by env)
    dist_to_gate_m: float         # dist from car to current target gate center
    gate_crossed: bool            # did the car cross the target gate this step?
    is_final_gate: bool           # is the current target gate the last one?
    num_gates: int                # total number of gates


@dataclass
class RewardResult:
    """Output of compute(). env.py applies these fields directly."""
    reward: float
    terminated: bool
    truncated: bool
    reason: str
    episode_state: EpisodeState


# ---------------------------------------------------------------------------
# 2D segment intersection — used by env.py, exposed here for convenience
# ---------------------------------------------------------------------------

def segments_intersect_2d(
    p1: tuple, p2: tuple,   # movement segment (prev_pos, cur_pos) as (x, z)
    q1: tuple, q2: tuple,   # gate segment (left_post, right_post) as (x, z)
) -> bool:
    """
    Check if two 2D line segments intersect.
    Uses the cross-product orientation test.
    """
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    d1 = cross(q1, q2, p1)
    d2 = cross(q1, q2, p2)
    d3 = cross(p1, p2, q1)
    d4 = cross(p1, p2, q2)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True

    # Collinear cases (treat as no intersection for gate purposes)
    return False


# ---------------------------------------------------------------------------
# Core reward function
# ---------------------------------------------------------------------------

_DEFAULT_CFG = RewardConfig()


def compute(
    ctx: RewardContext,
    state: EpisodeState,
    cfg: RewardConfig = _DEFAULT_CFG,
) -> RewardResult:
    """
    Compute reward + termination for one env step.

    Pure function of its arguments (no globals, no TM state).
    ``state`` is mutated in-place for efficiency and returned inside the result.
    """
    reward = 0.0
    terminated = False
    truncated = False
    reason = ""

    # Track max speed
    state.max_speed = max(state.max_speed, ctx.speed_kmh)

    # Update has_moved gate (prevents stuck from firing before car moves)
    if ctx.speed_kmh > cfg.move_threshold_kmh:
        state.has_moved = True

    # ==== GATE CROSSING (PRIMARY REWARD) ====

    if ctx.gate_crossed:
        r_gate = cfg.gate_reward
        reward += r_gate
        state.breakdown.gate_cross += r_gate
        state.gates_crossed += 1

        # Advance to next gate
        state.current_gate_idx += 1
        state.prev_dist_to_gate = float("inf")  # reset approach tracking
        state.consecutive_wrong_way = 0          # reset wrong-way counter

        # Check if finished (crossed final gate)
        if ctx.is_final_gate:
            reward += cfg.finish_bonus
            state.breakdown.finish += cfg.finish_bonus
            terminated = True
            reason = "finish_all_gates"

    # ==== DENSE SHAPING ====

    if not terminated:
        # Approach reward: positive when closing distance to next gate
        if state.prev_dist_to_gate < float("inf"):
            delta_dist = state.prev_dist_to_gate - ctx.dist_to_gate_m
            r_approach = delta_dist * cfg.approach_weight
            reward += r_approach
            state.breakdown.approach += r_approach
        state.prev_dist_to_gate = ctx.dist_to_gate_m

        # Speed reward
        r_speed = (ctx.speed_kmh / cfg.speed_max_kmh) * cfg.speed_weight
        reward += r_speed
        state.breakdown.speed += r_speed

        # Time penalty
        reward -= cfg.time_penalty
        state.breakdown.time_cost -= cfg.time_penalty

    # ==== TERMINATORS ====

    if not terminated:
        # Fell off track
        if ctx.position[1] < cfg.fell_off_y:
            reward -= cfg.fell_off_penalty
            state.breakdown.fell_off -= cfg.fell_off_penalty
            terminated = True
            reason = "fell_off_track"

    if not terminated:
        # Crash — sudden speed drop (wall collision)
        drop = ctx.prev_speed_kmh - ctx.speed_kmh
        if drop > cfg.crash_speed_drop_kmh:
            reward -= cfg.crash_penalty
            state.breakdown.crash -= cfg.crash_penalty
            terminated = True
            reason = "crash_wall_hit"

    if not terminated:
        # Stuck — only after has_moved gate
        if ctx.speed_kmh < cfg.stuck_speed_thresh:
            state.consecutive_stuck += 1
        else:
            state.consecutive_stuck = 0

        if state.has_moved and state.consecutive_stuck >= cfg.stuck_steps:
            reward -= cfg.stuck_penalty
            state.breakdown.stuck -= cfg.stuck_penalty
            terminated = True
            reason = "stuck_no_speed"

    if not terminated:
        # Wrong way — distance to gate keeps increasing
        if ctx.dist_to_gate_m > cfg.wrong_way_dist_m:
            state.consecutive_wrong_way += 1
        else:
            state.consecutive_wrong_way = 0

        if state.consecutive_wrong_way >= cfg.wrong_way_steps:
            reward -= cfg.wrong_way_penalty
            state.breakdown.wrong_way -= cfg.wrong_way_penalty
            terminated = True
            reason = "wrong_way"

    if not terminated:
        # Timeout → truncated
        if ctx.step_idx >= cfg.timeout_steps:
            truncated = True
            reason = "timeout"

    if not terminated and not truncated:
        # Startup stuck: car never moved at all
        if not state.has_moved and ctx.step_idx >= cfg.startup_stuck_steps:
            terminated = True
            reason = "startup_stuck"

    return RewardResult(
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        reason=reason,
        episode_state=state,
    )
