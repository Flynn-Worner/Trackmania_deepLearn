"""
LEGACY — Gymnasium environment for Trackmania Nations Forever via TMInterface sockets.

NOTE: This file is DEPRECATED.  Training now uses interfacing/env.py with the
gate-based reward system (data/gates.json).  This file is preserved for
reference only.  Do not import from training code.

Simulation speed
----------------
TRAINING_SPEED = 10 (10× faster than real-time).  Set immediately on connect
and restored to 10 after every tp warm-up.  Equivalent to typing
  set speed 10
in the TMInterface console.  Episodes finish ~10× faster in wall-clock time;
the physics are identical from the agent's perspective.

Centerline / spline
-------------------
Block centers from map_blocks.json are 32 m apart (one per grid cell).
For curves this puts the "centerline" point in the geometric middle of the
cell, which is NOT on the actual road surface — hence the car tries to drive
through walls.

Fix: _build_path_points() inserts face-midpoints between every consecutive
pair of block centers.  The face-midpoint is exactly where the road CROSSES
the boundary between two grid cells, so it is always ON the road even through
turns.  This halves the spacing to ~16 m and puts all points on the road.

_project_onto_path() then projects the car position onto the closest LINE
SEGMENT of the denser path rather than snapping to the nearest isolated point.
This gives:
  - Accurate dist_to_centerline for the observation and penalty
  - Smooth continuous arc-length (sub-block resolution) for the progress reward

Spawn strategies
----------------
  1. State-history rewind  (40% when ≥100 states saved)
  2. Random TP + warm-up   (40%, curriculum-gated by history size)
       execute_command("tp X Y Z")   – teleport to random track position
       set_speed(N)                  – fast-forward warmup acceleration
  3. Start-line restart     (20%, always available)
"""

import random
import math
import json
import os
import sys

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .tminterface2 import TMInterface, MessageType

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_processing.features import StateNormalizer


class TrackmaniaEnv(gym.Env):
    """
    Custom Gymnasium environment for TMNF.

    Observation space (6 floats, pre-scaled to ≈[-1,1] before VecNormalize):
        [speed_norm, dist_to_path_norm, yaw_norm, pitch_norm, roll_norm,
         checkpoints_norm]

    Action space – continuous Box(3,) in [-1, 1]:
        [steer, gas, brake]
        steer  : mapped linearly to TM integer range [-65536, 65536]
        gas    : > 0 → accelerate  (binary; game API limitation)
        brake  : > 0 → brake        (binary; game API limitation)
    """

    metadata = {"render_modes": ["human"]}

    # Simulation speed during normal training episodes (10× real-time).
    # Higher values train faster but require Python to keep up with socket I/O.
    # The reference_linesight project uses 80×; 10 is a safe starting value.
    TRAINING_SPEED = 10.0

    # Warm-up speed during tp-spawn acceleration phase.
    # Using the same value as TRAINING_SPEED keeps warm-up behaviour consistent.
    WARMUP_SPEED_MULT = 10.0

    # The stadium track surface in TMNF sits at approximately 26 m in world Y.
    # Spawning 1 m above avoids clipping through the road mesh.
    # Adjust if your map has a different surface height.
    TRACK_SURFACE_Y = 26.0

    def __init__(self, port: int = 8483, ticks_per_step: int = 25):
        super().__init__()

        self.port = port
        self.ticks_per_step = ticks_per_step

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )

        # ------------------------------------------------------------------
        # Load, sort and densify the path
        # ------------------------------------------------------------------
        self.map_blocks = []      # sorted list of block dicts (32 m spacing)
        self.path_points = []     # denser waypoints: block centres + face midpoints (~16 m)
        self.path_arc = []        # cumulative arc-lengths at each path_point

        blocks_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "map_blocks.json"
        )
        if os.path.exists(blocks_path):
            with open(blocks_path, "r") as f:
                raw_blocks = json.load(f)
            self.map_blocks = raw_blocks
            # Insert face-midpoints so the path closely follows the road
            # even through curves (details in _build_path_points docstring).
            self.path_points = self._build_path_points(self.map_blocks)
            self.path_arc = self._compute_arc_from_points(self.path_points)
            total_m = self.path_arc[-1] if self.path_arc else 0.0
            print(
                f"[Port {self.port}] {len(self.map_blocks)} blocks → "
                f"{len(self.path_points)} path points, "
                f"arc length ≈ {total_m:.0f} m"
            )
        else:
            print(f"[Port {self.port}] WARNING: data/map_blocks.json not found. "
                  "Run extract_spline.py first.")

        self.highest_arc_length = 0.0
        self.state_history = []

        self.iface = TMInterface(self.port)
        self.connected = False

        self.current_state = None
        self.previous_speed = 0.0
        self.consecutive_stuck_steps = 0

        self.normalizer = StateNormalizer()

    # ------------------------------------------------------------------
    # Path construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sort_blocks(raw_blocks):
        """Greedy nearest-neighbour sort: start block first, finish last."""
        if not raw_blocks:
            return []

        def is_start(b):
            return "start" in b["name"].lower()

        def is_finish(b):
            return "finish" in b["name"].lower() and "start" not in b["name"].lower()

        starts = [b for b in raw_blocks if is_start(b)]
        finishes = [b for b in raw_blocks if is_finish(b)]
        middles = [b for b in raw_blocks if not is_start(b) and not is_finish(b)]

        if not starts:
            starts = [raw_blocks[0]]
            middles = raw_blocks[1:]

        ordered = [starts[0]]
        remaining = middles + starts[1:]
        while remaining:
            last = ordered[-1]["world_center"]
            best_i, best_d = 0, float("inf")
            for i, b in enumerate(remaining):
                c = b["world_center"]
                d = (last["x"] - c["x"]) ** 2 + (last["z"] - c["z"]) ** 2
                if d < best_d:
                    best_d, best_i = d, i
            ordered.append(remaining.pop(best_i))

        ordered.extend(finishes)
        return ordered

    @staticmethod
    def _build_path_points(blocks):
        """
        Build a dense set of path waypoints from sorted block centres.

        Why face-midpoints fix curves
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        TMNF blocks occupy 32×32 m grid cells.  The geometric CENTRE of a
        cell sits in the middle of the grid square.  For straight blocks this
        is on the road.  For curved/corner blocks it is NOT — the road arc
        passes near the cell edges, not the cell centre.

        The midpoint between two consecutive block centres equals the midpoint
        of the SHARED FACE between those two cells, which is exactly where the
        road crosses from one block to the next.  That point IS always on the
        road regardless of block type.

        Result: alternating original centres + face midpoints, ~16 m spacing,
        all points lying on or very close to the actual road surface.

        Layout for blocks A → B → C:
          [A_centre, A-B_face, B_centre, B-C_face, C_centre]
        """
        if not blocks:
            return []

        pts = []
        for i, b in enumerate(blocks):
            c = b["world_center"]
            pts.append((c["x"], c["y"], c["z"]))
            if i < len(blocks) - 1:
                n = blocks[i + 1]["world_center"]
                # Face midpoint: boundary between block i and block i+1
                pts.append((
                    (c["x"] + n["x"]) / 2.0,
                    (c["y"] + n["y"]) / 2.0,
                    (c["z"] + n["z"]) / 2.0,
                ))
        return pts

    @staticmethod
    def _compute_arc_from_points(pts):
        """Cumulative 3-D arc-lengths for a list of (x, y, z) tuples."""
        if not pts:
            return []
        arc = [0.0]
        for i in range(1, len(pts)):
            ax, ay, az = pts[i - 1]
            bx, by, bz = pts[i]
            d = math.sqrt((bx - ax) ** 2 + (by - ay) ** 2 + (bz - az) ** 2)
            arc.append(arc[-1] + d)
        return arc

    def _project_onto_path(self, pos):
        """
        Project world position `pos` onto the nearest segment of path_points.

        Returns
        -------
        seg_idx   : int    index of the segment start in path_points
        dist      : float  perpendicular distance from pos to the nearest
                           point on that segment (metres)
        arc       : float  cumulative arc-length at the projected point
                           (sub-segment resolution — useful for smooth rewards)
        """
        px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
        min_dist = float("inf")
        best_seg = 0
        best_arc = 0.0

        for i in range(len(self.path_points) - 1):
            ax, ay, az = self.path_points[i]
            bx, by, bz = self.path_points[i + 1]

            dx, dy, dz = bx - ax, by - ay, bz - az
            seg_len_sq = dx * dx + dy * dy + dz * dz

            if seg_len_sq < 1e-9:
                t = 0.0
            else:
                t = ((px - ax) * dx + (py - ay) * dy + (pz - az) * dz) / seg_len_sq
                t = max(0.0, min(1.0, t))

            cx = ax + t * dx
            cy = ay + t * dy
            cz = az + t * dz
            d = math.sqrt((px - cx) ** 2 + (py - cy) ** 2 + (pz - cz) ** 2)

            if d < min_dist:
                min_dist = d
                best_seg = i
                seg_len = math.sqrt(seg_len_sq)
                best_arc = self.path_arc[i] + t * seg_len

        return best_seg, min_dist, best_arc

    def _update_highest_arc_from_state(self, state):
        if not self.path_points:
            return
        _, _, arc = self._project_onto_path(state.position)
        self.highest_arc_length = arc

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_observation(self, state):
        """
        Build a normalised 6-float observation.

          0  speed / 300               (0 → 1 at 300 km/h)
          1  dist_to_path / 50         (0 → 1 at 50 m off road)
          2  yaw  / π                  (-1 → 1)
          3  pitch / π                 (-1 → 1)
          4  roll  / π                 (-1 → 1)
          5  checkpoints / 20          (0 → 1)
        """
        speed = state.display_speed
        pos = state.position
        yaw, pitch, roll = state.yaw_pitch_roll
        checkpoints = state.cp_data.cp_times_length if state.cp_data else 0

        dist_to_path = 0.0
        if self.path_points:
            _, dist_to_path, _ = self._project_onto_path(pos)

        raw_obs = np.array([
            float(speed) / 300.0,
            float(dist_to_path) / 50.0,
            float(yaw) / math.pi,
            float(pitch) / math.pi,
            float(roll) / math.pi,
            float(checkpoints) / 20.0,
        ], dtype=np.float32)

        # Guard against NaN/Inf from bad game states (e.g. during race restart).
        # Without this, a single bad frame permanently corrupts VecNormalize's
        # running mean/std, causing the 'invalid value encountered' RuntimeWarning.
        raw_obs = np.nan_to_num(raw_obs, nan=0.0, posinf=5.0, neginf=-5.0)

        return self.normalizer.normalize(raw_obs)

    # ------------------------------------------------------------------
    # Action
    # ------------------------------------------------------------------

    def _apply_action(self, action):
        """
        Map continuous [-1,1]³ action to TMInterface inputs.
        Steer is continuous; gas/brake are binary (game API limitation).
        """
        steer = int(np.clip(float(action[0]), -1.0, 1.0) * 65536)
        accelerate = bool(float(action[1]) > 0.0)
        brake = bool(float(action[2]) > 0.0)
        self.iface.set_input_state(accelerate=accelerate, brake=brake, steer=steer)

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self):
        if not self.connected:
            print(f"[Port {self.port}] Connecting to TMInterface ...")
            self.iface.register(timeout=None)
            self.connected = True

            while True:
                msgtype = self.iface._read_int32()
                if msgtype == int(MessageType.SC_ON_CONNECT_SYNC):
                    self.iface.set_on_step_period(self.ticks_per_step)
                    # Run at TRAINING_SPEED× from the very first step so every
                    # episode (not just tp warm-up) benefits from fast simulation.
                    self.iface.set_speed(self.TRAINING_SPEED)
                    self.iface._respond_to_call(msgtype)
                    break
                else:
                    self.iface._respond_to_call(msgtype)

            print(f"[Port {self.port}] Connected — simulation speed {self.TRAINING_SPEED}×.")

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action):
        if not self.connected:
            self.connect()

        self._apply_action(action)

        terminated = False
        truncated = False
        reward = 0.0

        while True:
            msgtype = self.iface._read_int32()

            if msgtype == int(MessageType.SC_RUN_STEP_SYNC):
                _time = self.iface._read_int32()
                state = self.iface.get_simulation_state()

                # -------------------------------------------------------
                # REWARD
                #
                # At 150 km/h (~41.7 m/s, 10 Hz step rate):
                #   speed   = 150/300 × 0.5              ≈ +0.25/step
                #   progress= 4.17 m × 0.05              ≈ +0.21/step
                # Both signals are roughly equal so the agent must both go
                # fast AND advance along the track.
                # -------------------------------------------------------

                # 1. Speed reward (capped at 0.5/step at 300 km/h).
                reward += (state.display_speed / 300.0) * 0.5

                dist_to_path = 0.0
                if self.path_points:
                    pos = state.position
                    _, dist_to_path, current_arc = self._project_onto_path(pos)

                    # 2. Arc-length progress: reward each new metre of track.
                    if current_arc > self.highest_arc_length:
                        reward += (current_arc - self.highest_arc_length) * 0.05
                        self.highest_arc_length = current_arc

                    # 3. Centerline penalty (small nudge toward road centre).
                    reward -= dist_to_path * 0.003

                    # 4. Per-step time penalty.
                    reward -= 0.005

                    # 5. Fell off elevated track.
                    if pos[1] < 20.0:
                        reward -= 10.0
                        terminated = True

                # 6. Crash: sudden large speed drop (wall hit).
                speed_drop = self.previous_speed - state.display_speed
                if speed_drop > 60.0:
                    reward -= 10.0
                    terminated = True

                # 7. Stuck at near-zero speed.
                if state.display_speed < 10.0:
                    self.consecutive_stuck_steps += 1
                else:
                    self.consecutive_stuck_steps = 0
                if self.consecutive_stuck_steps >= 50:
                    reward -= 10.0
                    terminated = True

                self.previous_speed = state.display_speed

                if not terminated and state.display_speed > 20.0 and dist_to_path < 15.0:
                    if len(self.state_history) < 10000 and random.random() < 0.1:
                        self.state_history.append(state)

                obs = self._get_observation(state)
                self.current_state = state
                self.iface._respond_to_call(msgtype)
                break

            elif msgtype == int(MessageType.SC_CHECKPOINT_COUNT_CHANGED_SYNC):
                current = self.iface._read_int32()
                target = self.iface._read_int32()

                reward += 20.0
                if current == target:
                    terminated = True
                    reward += 50.0
                    print(f"[Port {self.port}] FINISH LINE REACHED!")

                self.iface._respond_to_call(msgtype)

            else:
                self.iface._respond_to_call(msgtype)

        return obs, reward, terminated, truncated, {}

    # ------------------------------------------------------------------
    # Spawn helpers
    # ------------------------------------------------------------------

    def _consume_one_run_step(self):
        """Block until the next SC_RUN_STEP_SYNC and return the SimStateData."""
        while True:
            mt = self.iface._read_int32()
            if mt == int(MessageType.SC_RUN_STEP_SYNC):
                self.iface._read_int32()
                state = self.iface.get_simulation_state()
                self.iface._respond_to_call(mt)
                return state
            else:
                self.iface._respond_to_call(mt)

    def _tp_warmup_spawn(self, n_history: int):
        """
        Teleport to a random block along the centreline, then accelerate for
        a random number of warm-up steps so the episode starts with varied speed.

        TMInterface commands used
        -------------------------
        tp X Y Z      : teleport car to world coordinates
                        (via execute_command)
        set_speed(N)  : run physics at N× (here = WARMUP_SPEED_MULT) during
                        warm-up, then restore to TRAINING_SPEED for the episode

        Curriculum gating
        -----------------
        With no history the car always spawns near the start.
        As state_history grows (target ~2000) the spawn window expands to
        cover 60% of the track so the agent practises later sections earlier.
        """
        n_blocks = len(self.map_blocks)
        progress_ratio = min(n_history / 2000.0, 1.0)
        max_idx = max(0, int((n_blocks - 1) * 0.6 * progress_ratio))
        spawn_idx = random.randint(0, max_idx) if max_idx > 0 else 0

        block = self.map_blocks[spawn_idx]
        x = block["world_center"]["x"]
        # Always use TRACK_SURFACE_Y + 1 m to avoid clipping through the road.
        # Block grid Y coordinates give the block base, not the road surface.
        y = self.TRACK_SURFACE_Y + 1.0
        z = block["world_center"]["z"]

        self.iface.execute_command(f"tp {x:.3f} {y:.3f} {z:.3f}")

        # 5–50 warm-up steps → ~0.5–5 s of game time → ~5–75 km/h starting speed.
        # Minimum 5 to ensure physics has processed the tp command.
        warmup_steps = random.randint(5, 50)

        # Speed up physics during warm-up so wall-clock cost is minimal.
        self.iface.set_speed(self.WARMUP_SPEED_MULT)
        state = None
        for _ in range(warmup_steps):
            self.iface.set_input_state(accelerate=True, brake=False, steer=0)
            state = self._consume_one_run_step()

        # Restore training speed (not real-time — stay at TRAINING_SPEED).
        self.iface.set_speed(self.TRAINING_SPEED)
        return state

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        """
        Three spawn strategies (chosen probabilistically each episode):

          40%  State-history rewind  – rewind to a real past state,
               preserving speed and orientation.  Requires ≥100 saved states.

          40%  TP + warm-up          – teleport to a random track position
               and accelerate for a random number of steps.

          20%  Start-line restart    – give_up(), always available.
        """
        super().reset(seed=seed)
        if not self.connected:
            self.connect()

        n_hist = len(self.state_history)
        roll = random.random()
        use_history = n_hist > 100 and roll < 0.40
        use_tp = self.map_blocks and (not use_history) and roll < 0.80

        if use_history:
            self.iface.rewind_to_state(random.choice(self.state_history))
            state = self._consume_one_run_step()
            self._update_highest_arc_from_state(state)
            self.previous_speed = state.display_speed
            self.consecutive_stuck_steps = 0
            obs = self._get_observation(state)
            self.current_state = state
            return obs, {}

        # Both tp-spawn and start-line need the race to restart first.
        self.iface.give_up()

        first_frame = True
        prev_time = -1
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

        if use_tp:
            state = self._tp_warmup_spawn(n_hist)
            self._update_highest_arc_from_state(state)
            self.previous_speed = state.display_speed
        else:
            self.highest_arc_length = 0.0
            self.previous_speed = 0.0

        self.consecutive_stuck_steps = 0
        obs = self._get_observation(state)
        self.current_state = state
        return obs, {}

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------

    def close(self):
        if self.connected:
            self.iface.close()
            self.connected = False
