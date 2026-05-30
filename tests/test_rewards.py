"""
Unit tests for interfacing/rewards.py

Run with:  python -m pytest tests/test_rewards.py -v
    or:    python tests/test_rewards.py

No TM connection required — all tests use fake RewardContext objects.
"""

from __future__ import annotations

import sys
import os
import unittest

import numpy as np

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interfacing.rewards import (
    RewardConfig, RewardContext, EpisodeState, compute,
)


def _make_ctx(**overrides) -> RewardContext:
    """Build a RewardContext with sensible defaults; override any field."""
    defaults = dict(
        position=np.array([100.0, 30.0, 200.0]),
        speed_kmh=80.0,
        prev_speed_kmh=80.0,
        dist_to_route_m=5.0,
        current_arc_m=100.0,
        highest_arc_m=90.0,
        heading_align=None,
        action=np.array([0.0, 1.0, 0.0]),
        prev_action=np.array([0.0, 1.0, 0.0]),
        step_idx=50,
        race_time_ms=5000,
        checkpoint_hit=False,
        is_finish=False,
        has_lateral_contact=False,
    )
    defaults.update(overrides)
    return RewardContext(**defaults)


class TestProgressReward(unittest.TestCase):
    """Arc-length progress should yield positive reward."""

    def test_forward_progress(self):
        cfg = RewardConfig(progress_weight=0.20)
        ctx = _make_ctx(current_arc_m=100.0, highest_arc_m=90.0, dist_to_route_m=5.0)
        state = EpisodeState(highest_arc_m=90.0)
        result = compute(ctx, state, cfg)

        # 10 m progress * 0.20 = 2.0 from progress alone
        self.assertGreater(result.episode_state.breakdown.progress, 0.0)
        self.assertAlmostEqual(result.episode_state.breakdown.progress, 10.0 * 0.20)
        # Cursor should advance
        self.assertAlmostEqual(result.episode_state.highest_arc_m, 100.0)

    def test_no_backtrack_reward(self):
        """Going backward should NOT yield progress reward."""
        ctx = _make_ctx(current_arc_m=80.0, highest_arc_m=100.0, dist_to_route_m=5.0)
        state = EpisodeState(highest_arc_m=100.0)
        result = compute(ctx, state)

        self.assertAlmostEqual(result.episode_state.breakdown.progress, 0.0)
        # Cursor stays at 100, not regressed to 80
        self.assertAlmostEqual(result.episode_state.highest_arc_m, 100.0)

    def test_progress_gated_by_distance(self):
        """Progress should NOT count when too far from route."""
        cfg = RewardConfig(progress_dist_gate_m=10.0)
        ctx = _make_ctx(current_arc_m=200.0, highest_arc_m=100.0, dist_to_route_m=50.0)
        state = EpisodeState(highest_arc_m=100.0)
        result = compute(ctx, state, cfg)

        self.assertAlmostEqual(result.episode_state.breakdown.progress, 0.0)
        self.assertAlmostEqual(result.episode_state.highest_arc_m, 100.0)


class TestStuckTermination(unittest.TestCase):
    """Stuck detection with has_moved grace."""

    def test_stuck_without_has_moved_no_terminate(self):
        """50 steps at 0 km/h but has_moved=False -> NOT terminated."""
        state = EpisodeState(has_moved=False, consecutive_stuck=49)
        ctx = _make_ctx(speed_kmh=0.0, prev_speed_kmh=0.0, step_idx=100)
        result = compute(ctx, state)

        # consecutive_stuck increments to 50, but has_moved is False
        self.assertFalse(result.terminated)
        self.assertEqual(result.reason, "")

    def test_stuck_after_has_moved_terminates(self):
        """50 steps at 0 km/h after has_moved=True -> terminated."""
        state = EpisodeState(has_moved=True, consecutive_stuck=49)
        ctx = _make_ctx(speed_kmh=0.0, prev_speed_kmh=0.0, step_idx=100)
        result = compute(ctx, state)

        self.assertTrue(result.terminated)
        self.assertEqual(result.reason, "stuck_no_speed")
        self.assertLess(result.reward, 0.0)

    def test_stuck_counter_resets_on_speed(self):
        """Moving above threshold resets stuck counter."""
        state = EpisodeState(has_moved=True, consecutive_stuck=30)
        ctx = _make_ctx(speed_kmh=50.0, prev_speed_kmh=50.0, step_idx=100)
        result = compute(ctx, state)

        self.assertEqual(result.episode_state.consecutive_stuck, 0)
        self.assertFalse(result.terminated)


class TestFellOff(unittest.TestCase):
    def test_fell_off_track(self):
        """Position Y < 20 -> terminated with fell_off_track."""
        ctx = _make_ctx(position=np.array([100.0, 15.0, 200.0]))
        state = EpisodeState()
        result = compute(ctx, state)

        self.assertTrue(result.terminated)
        self.assertEqual(result.reason, "fell_off_track")
        self.assertLess(result.episode_state.breakdown.fell_off, 0.0)


class TestCrash(unittest.TestCase):
    def test_crash_speed_drop(self):
        """Speed drop > 60 km/h -> crash termination."""
        ctx = _make_ctx(speed_kmh=20.0, prev_speed_kmh=100.0)
        state = EpisodeState()
        result = compute(ctx, state)

        self.assertTrue(result.terminated)
        self.assertEqual(result.reason, "crash_wall_hit")

    def test_normal_braking_no_crash(self):
        """Speed drop < 60 should not crash."""
        ctx = _make_ctx(speed_kmh=80.0, prev_speed_kmh=120.0)
        state = EpisodeState()
        result = compute(ctx, state)

        self.assertFalse(result.terminated)


class TestFinish(unittest.TestCase):
    def test_finish_line(self):
        """is_finish=True -> terminated with finish_line."""
        ctx = _make_ctx(is_finish=True, checkpoint_hit=True)
        state = EpisodeState()
        result = compute(ctx, state)

        self.assertTrue(result.terminated)
        self.assertEqual(result.reason, "finish_line")
        self.assertGreater(result.episode_state.breakdown.finish, 0.0)
        self.assertGreater(result.episode_state.breakdown.checkpoint, 0.0)


class TestTimeout(unittest.TestCase):
    def test_timeout_truncation(self):
        """step_idx >= 3000 -> truncated (NOT terminated)."""
        cfg = RewardConfig(timeout_steps=3000)
        ctx = _make_ctx(step_idx=3000)
        state = EpisodeState()
        result = compute(ctx, state, cfg)

        self.assertFalse(result.terminated)
        self.assertTrue(result.truncated)
        self.assertEqual(result.reason, "timeout")


class TestOffRoute(unittest.TestCase):
    def test_off_route_sustained(self):
        """dist > 40 m for 20 steps -> terminated."""
        cfg = RewardConfig(off_route_dist_m=40.0, off_route_steps=20)
        state = EpisodeState(consecutive_off_route=19)
        ctx = _make_ctx(dist_to_route_m=50.0, step_idx=100)
        result = compute(ctx, state, cfg)

        self.assertTrue(result.terminated)
        self.assertEqual(result.reason, "off_route")

    def test_off_route_resets_on_return(self):
        """Returning to route resets off-route counter."""
        state = EpisodeState(consecutive_off_route=15)
        ctx = _make_ctx(dist_to_route_m=5.0, step_idx=100)
        result = compute(ctx, state)

        self.assertEqual(result.episode_state.consecutive_off_route, 0)
        self.assertFalse(result.terminated)


class TestCheckpoint(unittest.TestCase):
    def test_checkpoint_bonus(self):
        """Hitting a checkpoint (non-finish) gives bonus without termination."""
        cfg = RewardConfig(checkpoint_bonus=20.0)
        ctx = _make_ctx(checkpoint_hit=True, is_finish=False)
        state = EpisodeState()
        result = compute(ctx, state, cfg)

        self.assertFalse(result.terminated)
        self.assertAlmostEqual(result.episode_state.breakdown.checkpoint, 20.0)


class TestSpeedReward(unittest.TestCase):
    def test_speed_scales_linearly(self):
        """Speed reward should be proportional to speed/max."""
        cfg = RewardConfig(speed_weight=0.05)
        ctx_fast = _make_ctx(speed_kmh=150.0)
        ctx_slow = _make_ctx(speed_kmh=50.0)
        s1, s2 = EpisodeState(), EpisodeState()
        r1 = compute(ctx_fast, s1, cfg)
        r2 = compute(ctx_slow, s2, cfg)

        self.assertGreater(r1.episode_state.breakdown.speed,
                           r2.episode_state.breakdown.speed)

    def test_zero_speed_zero_speed_reward(self):
        ctx = _make_ctx(speed_kmh=0.0)
        state = EpisodeState()
        result = compute(ctx, state)
        self.assertAlmostEqual(result.episode_state.breakdown.speed, 0.0)


class TestHasMovedGate(unittest.TestCase):
    def test_has_moved_set_on_speed(self):
        """has_moved becomes True when speed exceeds threshold."""
        state = EpisodeState(has_moved=False)
        ctx = _make_ctx(speed_kmh=20.0)
        result = compute(ctx, state)
        self.assertTrue(result.episode_state.has_moved)

    def test_has_moved_stays_false_at_low_speed(self):
        state = EpisodeState(has_moved=False)
        ctx = _make_ctx(speed_kmh=5.0)
        result = compute(ctx, state)
        self.assertFalse(result.episode_state.has_moved)


class TestNoRouteGraceful(unittest.TestCase):
    """Environment should work even without map_blocks.json."""

    def test_no_route_data(self):
        ctx = _make_ctx(dist_to_route_m=None, current_arc_m=None)
        state = EpisodeState()
        result = compute(ctx, state)

        # Should not crash; progress and centerline are 0
        self.assertAlmostEqual(result.episode_state.breakdown.progress, 0.0)
        self.assertAlmostEqual(result.episode_state.breakdown.centerline, 0.0)
        self.assertFalse(result.terminated)


class TestWallContact(unittest.TestCase):
    """Wall contact penalties and termination."""

    def test_per_step_penalty(self):
        """Touching the wall should yield a wall contact penalty."""
        cfg = RewardConfig(wall_contact_penalty=0.2)
        ctx = _make_ctx(has_lateral_contact=True)
        state = EpisodeState()
        result = compute(ctx, state, cfg)

        self.assertAlmostEqual(result.episode_state.breakdown.wall_contact, -0.2)
        self.assertEqual(result.episode_state.consecutive_wall_contact, 1)
        self.assertFalse(result.terminated)

    def test_counter_resets_on_no_contact(self):
        """Counter should reset to 0 when wall contact ends."""
        state = EpisodeState(consecutive_wall_contact=5)
        ctx = _make_ctx(has_lateral_contact=False)
        result = compute(ctx, state)

        self.assertEqual(result.episode_state.consecutive_wall_contact, 0)
        self.assertFalse(result.terminated)

    def test_termination_after_limit(self):
        """Consecutive steps at limit triggers termination and penalty."""
        cfg = RewardConfig(
            wall_contact_penalty=0.2,
            max_consecutive_wall_contact_steps=10,
            wall_contact_penalty_total=5.0,
        )
        state = EpisodeState(consecutive_wall_contact=9)
        ctx = _make_ctx(has_lateral_contact=True)
        result = compute(ctx, state, cfg)

        self.assertTrue(result.terminated)
        self.assertEqual(result.reason, "wall_contact")
        # -0.2 (step penalty) - 5.0 (total penalty) = -5.2
        self.assertAlmostEqual(result.episode_state.breakdown.wall_contact, -5.2)


if __name__ == "__main__":
    unittest.main()
