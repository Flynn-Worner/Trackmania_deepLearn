"""Reward logic for the gate-based TMNF environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import math

import numpy as np


def _orientation(a, b, c) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a, b, c) -> bool:
    return (
        min(a[0], c[0]) <= b[0] <= max(a[0], c[0])
        and min(a[1], c[1]) <= b[1] <= max(a[1], c[1])
    )


def segments_intersect_2d(p1, p2, q1, q2) -> bool:
    """Return True when two 2D line segments intersect."""
    o1 = _orientation(p1, p2, q1)
    o2 = _orientation(p1, p2, q2)
    o3 = _orientation(q1, q2, p1)
    o4 = _orientation(q1, q2, p2)

    if (o1 > 0 > o2 or o1 < 0 < o2) and (o3 > 0 > o4 or o3 < 0 < o4):
        return True

    eps = 1e-9
    if abs(o1) <= eps and _on_segment(p1, q1, p2):
        return True
    if abs(o2) <= eps and _on_segment(p1, q2, p2):
        return True
    if abs(o3) <= eps and _on_segment(q1, p1, q2):
        return True
    if abs(o4) <= eps and _on_segment(q1, p2, q2):
        return True
    return False


def gate_crossed_with_corridor(prev_pos, cur_pos, gate, gate_y_tolerance: float, gate_lateral_margin_m: float) -> bool:
    """Return True when the car crosses the gate plane inside a widened corridor.

    This is less angle-sensitive than a thin segment intersection: we care about
    the actual crossing point on the gate plane, not just the segment endpoints.
    """
    center = gate["center"]
    forward = gate["forward"]

    if abs(float(cur_pos[1]) - float(center[1])) > gate_y_tolerance:
        return False

    # Gate plane normal in XZ is the recorded forward vector.
    nx = float(forward[0])
    nz = float(forward[1])
    if abs(nx) < 1e-9 and abs(nz) < 1e-9:
        return False

    # Gate width axis is perpendicular to the normal.
    tx = nz
    tz = -nx

    prev_dx = float(prev_pos[0]) - float(center[0])
    prev_dz = float(prev_pos[2]) - float(center[2])
    cur_dx = float(cur_pos[0]) - float(center[0])
    cur_dz = float(cur_pos[2]) - float(center[2])

    prev_forward = prev_dx * nx + prev_dz * nz
    cur_forward = cur_dx * nx + cur_dz * nz

    # Must move through the gate plane from one side to the other.
    if prev_forward == cur_forward:
        return False
    if not ((prev_forward <= 0.0 <= cur_forward) or (prev_forward >= 0.0 >= cur_forward)):
        return False

    denom = prev_forward - cur_forward
    if abs(denom) < 1e-9:
        return False

    # Find where the car segment intersects the gate plane.
    t = prev_forward / denom
    if t < 0.0 or t > 1.0:
        return False

    ix = float(prev_pos[0]) + (float(cur_pos[0]) - float(prev_pos[0])) * t - float(center[0])
    iz = float(prev_pos[2]) + (float(cur_pos[2]) - float(prev_pos[2])) * t - float(center[2])

    # Keep the crossing inside the lateral corridor around the gate center.
    # We allow a small margin beyond the recorded width so shallow angles count.
    gate_left = gate["left_post"]
    gate_half_width = math.hypot(
        float(gate_left[0]) - float(center[0]),
        float(gate_left[2]) - float(center[2]),
    )
    corridor_half_width = gate_half_width + float(gate_lateral_margin_m)

    intersection_lateral = abs(ix * tx + iz * tz)
    if intersection_lateral > corridor_half_width:
        return False

    return True


@dataclass
class RewardBreakdown:
    gate_cross: float = 0.0
    approach: float = 0.0
    speed: float = 0.0
    braking: float = 0.0
    time_cost: float = 0.0
    finish: float = 0.0
    penalties: float = 0.0
    fell_off: float = 0.0
    wall_hit: float = 0.0
    wall_proximity: float = 0.0
    crash: float = 0.0
    stuck: float = 0.0
    wrong_way: float = 0.0
    total: float = 0.0


@dataclass
class RewardConfig:
    gate_y_tolerance: float = 7.5
    # Expand effective gate span a bit beyond recorded posts so edge-line passes count.
    gate_span_scale: float = 1.15
    gate_min_overhang_m: float = 2.0
    # Extra corridor width around the recorded gate posts for shallow-angle passes.
    gate_lateral_margin_m: float = 8.0
    # Fallback gate hit radius around gate center so non-square entries still count.
    gate_proximity_radius_m: float = 6.0
    gate_cross_reward: float = 12.0
    # Make heading toward the expected gate matter more.
    approach_scale: float = 0.35
    approach_clip: float = 3.0
    # Keep speed shaping small so it doesn't outweigh gate progression.
    speed_scale: float = 0.0005
    # Reward controlled deceleration near gates so the policy can set up corners.
    brake_near_gate_dist_m: float = 30.0
    brake_min_speed_kmh: float = 45.0
    brake_decel_min_kmh: float = 2.0
    brake_reward_scale: float = 0.02
    brake_reward_cap: float = 0.8
    time_penalty: float = -0.10
    finish_reward: float = 2500.0
    crash_speed_drop_kmh: float = 80.0
    crash_speed_floor_kmh: float = 25.0
    crash_penalty: float = -2.0
    # Penalize sharp unplanned speed drops (typically wall hits) with a small penalty.
    wall_hit_speed_drop_kmh: float = 18.0
    wall_hit_no_brake_threshold: float = 0.2
    wall_hit_penalty_scale: float = 0.12
    wall_hit_penalty_cap: float = 6.0
    # Smooth wall-proximity penalty ramp inside this distance.
    wall_proximity_threshold_m: float = 0.6
    wall_proximity_penalty_max: float = 1.0
    wall_proximity_power: float = 2.0
    min_track_y: float = 15.0
    fell_off_penalty: float = -20.0
    moved_speed_threshold_kmh: float = 25.0
    stuck_speed_threshold_kmh: float = 8.0
    stuck_steps: int = 80
    stuck_penalty: float = -10.0
    # If distance-to-target barely changes for too long at low speed,
    # treat it as wall-scrape/no-progress and restart.
    no_progress_delta_m: float = 0.20
    no_progress_speed_ceiling_kmh: float = 45.0
    no_progress_steps: int = 100
    wrong_way_margin_m: float = 1.5
    wrong_way_steps: int = 40
    wrong_way_penalty: float = -8.0
    max_steps: int = 4000
    timeout_penalty: float = -5.0


@dataclass
class RewardContext:
    position: np.ndarray
    prev_position: np.ndarray
    speed_kmh: float
    prev_speed_kmh: float
    action: np.ndarray
    prev_action: np.ndarray
    step_idx: int
    race_time_ms: int
    dist_to_gate_m: float
    dist_to_wall_m: float
    gate_crossed: bool
    is_final_gate: bool
    num_gates: int


@dataclass
class EpisodeState:
    current_gate_idx: int = 0
    gates_crossed: int = 0
    max_speed: float = 0.0
    consecutive_stuck: int = 0
    consecutive_no_progress: int = 0
    consecutive_wrong_way: int = 0
    has_moved: bool = False
    last_dist_to_gate_m: Optional[float] = None
    breakdown: RewardBreakdown = field(default_factory=RewardBreakdown)


@dataclass
class RewardResult:
    reward: float
    terminated: bool
    truncated: bool
    reason: str
    episode_state: EpisodeState


def compute(ctx: RewardContext, episode_state: EpisodeState, cfg: RewardConfig) -> RewardResult:
    breakdown = RewardBreakdown()
    next_state = EpisodeState(
        current_gate_idx=episode_state.current_gate_idx,
        gates_crossed=episode_state.gates_crossed,
        max_speed=max(episode_state.max_speed, float(ctx.speed_kmh)),
        consecutive_stuck=episode_state.consecutive_stuck,
        consecutive_no_progress=episode_state.consecutive_no_progress,
        consecutive_wrong_way=episode_state.consecutive_wrong_way,
        has_moved=(
            episode_state.has_moved
            or episode_state.gates_crossed > 0
            or bool(ctx.gate_crossed)
            or float(ctx.speed_kmh) >= cfg.moved_speed_threshold_kmh
        ),
        last_dist_to_gate_m=episode_state.last_dist_to_gate_m,
        breakdown=breakdown,
    )

    terminated = False
    truncated = False
    reason = "running"

    breakdown.time_cost = cfg.time_penalty

    if next_state.last_dist_to_gate_m is not None:
        delta = next_state.last_dist_to_gate_m - float(ctx.dist_to_gate_m)
        delta = float(np.clip(delta, -cfg.approach_clip, cfg.approach_clip))
        breakdown.approach = delta * cfg.approach_scale

        if next_state.has_moved and delta < -cfg.wrong_way_margin_m:
            next_state.consecutive_wrong_way += 1
        else:
            next_state.consecutive_wrong_way = 0

        # Catch low-speed wall scraping/oscillation: movement without progress.
        if (
            next_state.has_moved
            and not ctx.gate_crossed
            and abs(delta) < cfg.no_progress_delta_m
            and float(ctx.speed_kmh) <= cfg.no_progress_speed_ceiling_kmh
        ):
            next_state.consecutive_no_progress += 1
        else:
            next_state.consecutive_no_progress = 0

    next_state.last_dist_to_gate_m = float(ctx.dist_to_gate_m)

    breakdown.speed = max(0.0, float(ctx.speed_kmh)) * cfg.speed_scale

    if ctx.gate_crossed:
        breakdown.gate_cross = cfg.gate_cross_reward
        next_state.gates_crossed += 1
        next_state.current_gate_idx += 1
        next_state.consecutive_wrong_way = 0
        next_state.consecutive_stuck = 0
        next_state.consecutive_no_progress = 0
        next_state.last_dist_to_gate_m = None

        if ctx.is_final_gate or next_state.current_gate_idx >= ctx.num_gates:
            breakdown.finish = cfg.finish_reward
            terminated = True
            reason = "finish"

    speed_drop = float(ctx.prev_speed_kmh) - float(ctx.speed_kmh)
    brake_cmd = float(ctx.action[2]) > cfg.wall_hit_no_brake_threshold
    near_gate = float(ctx.dist_to_gate_m) <= cfg.brake_near_gate_dist_m

    # Controlled braking reward (supports safer corner entry).
    if (
        near_gate
        and brake_cmd
        and float(ctx.speed_kmh) >= cfg.brake_min_speed_kmh
        and speed_drop > cfg.brake_decel_min_kmh
    ):
        shaped_drop = speed_drop - cfg.brake_decel_min_kmh
        breakdown.braking = min(
            shaped_drop * cfg.brake_reward_scale,
            cfg.brake_reward_cap,
        )

    # Small wall-hit proxy penalty: large speed drop while not braking.
    if speed_drop >= cfg.wall_hit_speed_drop_kmh and not brake_cmd:
        drop_excess = speed_drop - cfg.wall_hit_speed_drop_kmh
        breakdown.wall_hit = -min(
            drop_excess * cfg.wall_hit_penalty_scale,
            cfg.wall_hit_penalty_cap,
        )

    # Soft ramp: once within 1 m of wall, increase penalty smoothly as distance shrinks.
    if float(ctx.dist_to_wall_m) < cfg.wall_proximity_threshold_m:
        closeness = 1.0 - float(ctx.dist_to_wall_m) / max(cfg.wall_proximity_threshold_m, 1e-6)
        closeness = float(np.clip(closeness, 0.0, 1.0))
        breakdown.wall_proximity = -cfg.wall_proximity_penalty_max * (closeness ** cfg.wall_proximity_power)

    if speed_drop >= cfg.crash_speed_drop_kmh and float(ctx.speed_kmh) <= cfg.crash_speed_floor_kmh:
        breakdown.crash = cfg.crash_penalty

    if float(ctx.position[1]) < cfg.min_track_y:
        breakdown.fell_off = cfg.fell_off_penalty
        terminated = True
        reason = "fell_off"

    if next_state.has_moved and not terminated:
        if float(ctx.speed_kmh) <= cfg.stuck_speed_threshold_kmh:
            next_state.consecutive_stuck += 1
        else:
            next_state.consecutive_stuck = 0

        if next_state.consecutive_stuck >= cfg.stuck_steps:
            breakdown.stuck = cfg.stuck_penalty
            terminated = True
            reason = "stuck"

        if next_state.consecutive_no_progress >= cfg.no_progress_steps:
            breakdown.stuck = cfg.stuck_penalty
            terminated = True
            reason = "stuck"

        if next_state.consecutive_wrong_way >= cfg.wrong_way_steps:
            breakdown.wrong_way = cfg.wrong_way_penalty
            terminated = True
            reason = "wrong_way"

    if not terminated and ctx.step_idx >= cfg.max_steps:
        breakdown.penalties += cfg.timeout_penalty
        truncated = True
        reason = "timeout"

    breakdown.penalties += (
        breakdown.fell_off
        + breakdown.wall_hit
        + breakdown.wall_proximity
        + breakdown.crash
        + breakdown.stuck
        + breakdown.wrong_way
    )
    breakdown.total = (
        breakdown.gate_cross
        + breakdown.approach
        + breakdown.speed
        + breakdown.braking
        + breakdown.time_cost
        + breakdown.finish
        + breakdown.penalties
    )

    next_state.breakdown = breakdown
    return RewardResult(
        reward=breakdown.total,
        terminated=terminated,
        truncated=truncated,
        reason=reason,
        episode_state=next_state,
    )