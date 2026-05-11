"""
Gymnasium environment for Trackmania Nations Forever via TMInterface sockets.

Spawn strategies used in reset():
  1. State-history rewind  (40% when ≥100 states saved)
     Rewinds physics to a real recorded state → preserves exact speed +
     orientation so the agent sees diverse situations from its own history.

  2. Random-TP + warm-up   (40%, curriculum-gated by history size)
     Uses TMInterface commands:
       execute_command("tp X Y Z")    – teleport car to random track block
       set_speed(N)                   – run physics at N× while accelerating
                                        to build varied initial speeds cheaply
     set_speed is the simulation-speed multiplier (e.g. 5 = 5× real-time).
     It is NOT the car velocity; the car accelerates naturally under gas.
     After warm-up, set_speed(1) restores real-time for the episode.

  3. Start-line restart     (remaining 20%, always available as fallback)
     give_up() restarts race from the start block at zero speed.
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
        [speed_norm, dist_to_center_norm, yaw_norm, pitch_norm, roll_norm,
         checkpoints_norm]

    Action space – continuous Box(3,) in [-1, 1]:
        [steer, gas, brake]
        steer  : mapped linearly to TM integer range [-65536, 65536]
        gas    : > 0 → accelerate  (binary; game API limitation)
        brake  : > 0 → brake        (binary; game API limitation)
    """

    metadata = {"render_modes": ["human"]}

    # Warm-up speed multiplier used during tp-spawn acceleration phase.
    # 5× means the car reaches driving speed ~5× faster in wall-clock time.
    WARMUP_SPEED_MULT = 5.0

    # The stadium track surface in TMNF sits at approximately 26 m in world Y.
    # Spawning 1 m above it avoids the car clipping through the road mesh.
    # Increase this constant if the car spawns underground on your map.
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
        # Load and path-sort map blocks
        # ------------------------------------------------------------------
        self.map_blocks = []
        self.arc_lengths = []

        blocks_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "map_blocks.json"
        )
        if os.path.exists(blocks_path):
            with open(blocks_path, "r") as f:
                raw_blocks = json.load(f)
            self.map_blocks = self._sort_blocks(raw_blocks)
            self.arc_lengths = self._compute_arc_lengths(self.map_blocks)
            total_m = self.arc_lengths[-1] if self.arc_lengths else 0.0
            print(
                f"[Port {self.port}] {len(self.map_blocks)} map blocks loaded, "
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
    # Path helpers (identical to extract_spline.py logic so the in-memory
    # sorted list matches what's written to disk)
    # ------------------------------------------------------------------

    @staticmethod
    def _sort_blocks(raw_blocks):
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
                d = math.sqrt((last["x"] - c["x"]) ** 2 + (last["z"] - c["z"]) ** 2)
                if d < best_d:
                    best_d, best_i = d, i
            ordered.append(remaining.pop(best_i))

        # Re-append finish block(s) at the end
        ordered.extend(finishes)
        return ordered

    @staticmethod
    def _compute_arc_lengths(blocks):
        if not blocks:
            return []
        arc = [0.0]
        for i in range(1, len(blocks)):
            prev = blocks[i - 1]["world_center"]
            cur = blocks[i]["world_center"]
            d = math.sqrt(
                (cur["x"] - prev["x"]) ** 2
                + (cur["y"] - prev["y"]) ** 2
                + (cur["z"] - prev["z"]) ** 2
            )
            arc.append(arc[-1] + d)
        return arc

    def _closest_block_idx(self, pos):
        min_dist = float("inf")
        closest_idx = 0
        for i, b in enumerate(self.map_blocks):
            c = b["world_center"]
            d = math.sqrt(
                (pos[0] - c["x"]) ** 2
                + (pos[1] - c["y"]) ** 2
                + (pos[2] - c["z"]) ** 2
            )
            if d < min_dist:
                min_dist = d
                closest_idx = i
        return closest_idx, min_dist

    def _update_highest_arc_from_state(self, state):
        if not self.map_blocks:
            return
        idx, _ = self._closest_block_idx(state.position)
        self.highest_arc_length = self.arc_lengths[idx]

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_observation(self, state):
        """
        Build a normalised 6-float observation.
        Each feature is pre-scaled to ≈[-1,1]/[0,1] before the StateNormalizer
        clip so VecNormalize's running statistics start in a sensible range.

          0  speed / 300          (0 → 1 at 300 km/h)
          1  dist_to_center / 50  (0 → 1 at 50 m off track)
          2  yaw  / π             (-1 → 1)
          3  pitch / π            (-1 → 1)
          4  roll  / π            (-1 → 1)
          5  checkpoints / 20     (0 → 1 for most maps)
        """
        speed = state.display_speed
        pos = state.position
        yaw, pitch, roll = state.yaw_pitch_roll
        checkpoints = state.cp_data.cp_times_length if state.cp_data else 0

        min_dist = 0.0
        if self.map_blocks:
            _, min_dist = self._closest_block_idx(pos)

        raw_obs = np.array([
            float(speed) / 300.0,
            float(min_dist) / 50.0,
            float(yaw) / math.pi,
            float(pitch) / math.pi,
            float(roll) / math.pi,
            float(checkpoints) / 20.0,
        ], dtype=np.float32)

        return self.normalizer.normalize(raw_obs)

    # ------------------------------------------------------------------
    # Action
    # ------------------------------------------------------------------

    def _apply_action(self, action):
        """
        Map a continuous [-1,1]³ action to TMInterface inputs.

        Steering is fully continuous: [-1, 1] → [-65536, 65536] (integer).
        Gas/brake are binary because TM's SetInputState only accepts 0/1.
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
                    self.iface._respond_to_call(msgtype)
                    break
                else:
                    self.iface._respond_to_call(msgtype)

            print(f"[Port {self.port}] Connected and synced.")

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action):
        if not self.connected:
            self.connect()

        self._apply_action(action)

        terminated = False
        truncated = False
        # Reward is accumulated (+= throughout), never reset mid-loop.
        # This prevents earlier checkpoint bonuses from being wiped by the
        # speed term that follows.
        reward = 0.0

        while True:
            msgtype = self.iface._read_int32()

            if msgtype == int(MessageType.SC_RUN_STEP_SYNC):
                _time = self.iface._read_int32()
                state = self.iface.get_simulation_state()

                # ----------------------------------------------------------
                # REWARD DESIGN RATIONALE
                # At 150 km/h (~41.7 m/s, ticks_per_step=25 → ~10 Hz):
                #   speed reward  = 150/300 * 0.5           = +0.25 / step
                #   progress      = 4.17 m * 0.05           = +0.21 / step
                # The two signals are now roughly equal so the agent must
                # actually advance along the track, not just spin in place.
                # ----------------------------------------------------------

                # 1. Speed reward – encourages driving fast, capped at 0.5/step.
                reward += (state.display_speed / 300.0) * 0.5

                dist_to_center = 0.0
                if self.map_blocks:
                    pos = state.position
                    closest_idx, dist_to_center = self._closest_block_idx(pos)

                    # 2. Arc-length progress – reward each new metre of track
                    #    conquered.  Only fires when reaching a new maximum so
                    #    driving backwards gives nothing.
                    current_arc = self.arc_lengths[closest_idx]
                    if current_arc > self.highest_arc_length:
                        reward += (current_arc - self.highest_arc_length) * 0.05
                        self.highest_arc_length = current_arc

                    # 3. Centerline penalty – kept small; just nudges the car
                    #    toward the middle rather than hugging walls.
                    reward -= dist_to_center * 0.003

                    # 4. Per-step time penalty – prefer finishing quickly.
                    reward -= 0.005

                    # 5. Fell off elevated track (Y well below surface).
                    if pos[1] < 20.0:
                        reward -= 10.0
                        terminated = True

                # 6. Crash: large sudden speed drop from hitting a wall.
                #    Threshold is 60 km/h (was 50) to avoid false positives
                #    during tp-spawn landings at lower speeds.
                speed_drop = self.previous_speed - state.display_speed
                if speed_drop > 60.0:
                    reward -= 10.0
                    terminated = True

                # 7. Stuck: near-zero speed for too long.
                if state.display_speed < 10.0:
                    self.consecutive_stuck_steps += 1
                else:
                    self.consecutive_stuck_steps = 0
                if self.consecutive_stuck_steps >= 50:
                    reward -= 10.0
                    terminated = True

                self.previous_speed = state.display_speed

                if not terminated and state.display_speed > 20.0 and dist_to_center < 15.0:
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
                self.iface._read_int32()           # discard race time
                state = self.iface.get_simulation_state()
                self.iface._respond_to_call(mt)
                return state
            else:
                self.iface._respond_to_call(mt)

    def _tp_warmup_spawn(self, n_history: int) -> "SimStateData":
        """
        Teleport the car to a random block along the centreline and accelerate
        it for a random number of steps so the episode begins with varied speed.

        TMInterface commands used:
          tp X Y Z   – teleports car to world coordinates (via execute_command)
          set_speed  – simulation-speed multiplier for fast warm-up
                       (equivalent to 'set speed N' in the TMInterface console)

        Curriculum: `n_history` gates how far along the track we may spawn.
        With no history the car always spawns at or near the start.  As history
        fills up (max ~2000 entries) the spawn window expands to 60% of the
        track length.
        """
        n_blocks = len(self.map_blocks)
        # Expand spawn window as the agent accumulates experience
        progress_ratio = min(n_history / 2000.0, 1.0)
        max_idx = max(0, int((n_blocks - 1) * 0.6 * progress_ratio))
        spawn_idx = random.randint(0, max_idx) if max_idx > 0 else 0

        block = self.map_blocks[spawn_idx]
        x = block["world_center"]["x"]
        # Always spawn 1 m above the known track surface height rather than
        # trying to derive Y from block grid coords (which gives the geometric
        # centre of the block, not the driveable surface).
        # TRACK_SURFACE_Y ≈ 26 m in the default TMNF stadium environment.
        y = self.TRACK_SURFACE_Y + 1.0
        z = block["world_center"]["z"]

        # 'tp X Y Z' is a standard TMInterface console command.
        # The car keeps its current orientation (facing direction from start
        # line), which adds orientation diversity to the curriculum.
        self.iface.execute_command(f"tp {x:.3f} {y:.3f} {z:.3f}")

        # Warm-up steps: 1–25 steps (minimum 1 so physics processes the tp).
        # At the default ticks_per_step=25 (≈10 Hz) this gives ≈0.1–2.5 s of
        # acceleration, producing initial speeds of roughly 5–75 km/h.
        warmup_steps = random.randint(1, 25)

        # Run physics at WARMUP_SPEED_MULT× so warm-up is cheap wall-clock time.
        # Equivalent to typing 'set speed 5' in the TMInterface console.
        self.iface.set_speed(self.WARMUP_SPEED_MULT)

        state = None
        for _ in range(warmup_steps):
            # Full gas, no steering so the car accelerates straight ahead.
            self.iface.set_input_state(accelerate=True, brake=False, steer=0)
            state = self._consume_one_run_step()

        # Restore real-time for the episode.
        self.iface.set_speed(1.0)
        return state

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        """
        Three spawn strategies, chosen probabilistically each episode:

          40%  State-history rewind  – fast recovery to a real past position.
               Requires ≥100 saved states; falls through to tp-spawn otherwise.

          40%  TP + warm-up spawn    – teleport to a random track block then
               accelerate for a random number of steps to give varied speed.
               Requires map_blocks to be loaded.

          20%  Start-line restart   – standard give_up(), always available.
        """
        super().reset(seed=seed)
        if not self.connected:
            self.connect()

        n_hist = len(self.state_history)
        roll = random.random()
        use_history = n_hist > 100 and roll < 0.40
        use_tp = self.map_blocks and (not use_history) and roll < 0.80

        if use_history:
            # ----------------------------------------------------------------
            # Strategy 1: rewind to a random preserved state
            # ----------------------------------------------------------------
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

        # Wait for the race timer to reset (time == 0 or a backwards jump).
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

                race_restarted = (_time == 0) or (_time > 0 and _time < prev_time - 100)
                if race_restarted:
                    break
                prev_time = _time
            else:
                self.iface._respond_to_call(msgtype)

        if use_tp:
            # ----------------------------------------------------------------
            # Strategy 2: teleport to a random block + warm-up
            # ----------------------------------------------------------------
            state = self._tp_warmup_spawn(n_hist)
            self._update_highest_arc_from_state(state)
            self.previous_speed = state.display_speed
        else:
            # ----------------------------------------------------------------
            # Strategy 3: start from the start line
            # ----------------------------------------------------------------
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
