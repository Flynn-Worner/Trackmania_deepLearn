import time
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .tminterface2 import TMInterface, MessageType

import sys
import os
import json
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_processing.features import StateNormalizer


class TrackmaniaEnv(gym.Env):
    """
    Gymnasium environment for Trackmania Nations Forever via TMInterface sockets.

    Observation space (6 floats, pre-scaled to roughly [-1, 1] before VecNormalize):
        [speed_norm, dist_to_center_norm, yaw_norm, pitch_norm, roll_norm, checkpoints_norm]

    Action space (continuous Box(3,) in [-1, 1]):
        [steer, gas, brake]
        - steer  : linearly mapped to TM integer range [-65536, 65536]
        - gas    : binary threshold at 0 (> 0 → accelerate)
        - brake  : binary threshold at 0 (> 0 → brake)
        Gas/brake are binary because TM's SetInputState API does not support
        analog throttle/brake magnitudes.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, port: int = 8483, ticks_per_step: int = 25):
        super().__init__()

        self.port = port
        self.ticks_per_step = ticks_per_step

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        # ------------------------------------------------------------------
        # Load and path-sort map blocks
        # ------------------------------------------------------------------
        self.map_blocks = []    # ordered list of block dicts
        self.arc_lengths = []   # cumulative metres along path for each block

        blocks_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "map_blocks.json"
        )
        if os.path.exists(blocks_path):
            with open(blocks_path, "r") as f:
                raw_blocks = json.load(f)

            self.map_blocks = self._sort_blocks(raw_blocks)
            self.arc_lengths = self._compute_arc_lengths(self.map_blocks)
            print(
                f"[Port {self.port}] Loaded {len(self.map_blocks)} map blocks, "
                f"total arc length {self.arc_lengths[-1]:.0f} m"
                if self.arc_lengths else f"[Port {self.port}] Loaded 0 map blocks"
            )
        else:
            print(f"[Port {self.port}] WARNING: data/map_blocks.json not found – "
                  "run extract_spline.py first.")

        # Track how far along the path we've reached (arc-length in metres)
        self.highest_arc_length = 0.0
        self.state_history = []

        self.iface = TMInterface(self.port)
        self.connected = False

        self.current_state = None
        self.previous_speed = 0.0
        self.consecutive_stuck_steps = 0

        self.normalizer = StateNormalizer()

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sort_blocks(raw_blocks):
        """
        Return blocks in track order: start block first, finish block last,
        road blocks in between via greedy nearest-neighbour in 2-D (X-Z).
        """
        if not raw_blocks:
            return []

        def is_start(b):
            return "start" in b["name"].lower()

        def is_finish(b):
            return "finish" in b["name"].lower()

        starts = [b for b in raw_blocks if is_start(b)]
        finishes = [b for b in raw_blocks if is_finish(b) and not is_start(b)]
        middles = [b for b in raw_blocks if not is_start(b) and not is_finish(b)]

        if not starts:
            starts = [raw_blocks[0]]
            middles = raw_blocks[1:]

        ordered = [starts[0]]
        remaining = middles + (finishes if finishes else [])

        while remaining:
            last = ordered[-1]["world_center"]
            best_i, best_d = 0, float("inf")
            for i, b in enumerate(remaining):
                c = b["world_center"]
                d = math.sqrt((last["x"] - c["x"]) ** 2 + (last["z"] - c["z"]) ** 2)
                if d < best_d:
                    best_d, best_i = d, i
            ordered.append(remaining.pop(best_i))

        # If finishes were mixed into remaining they sort naturally; if a finish
        # block ended up in the middle (rare), move it to the end.
        finish_indices = [i for i, b in enumerate(ordered) if is_finish(b) and i != len(ordered) - 1]
        for i in sorted(finish_indices, reverse=True):
            ordered.append(ordered.pop(i))

        return ordered

    @staticmethod
    def _compute_arc_lengths(blocks):
        """Cumulative 3-D Euclidean distance along the sorted block list."""
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
        """Return the index of the nearest block in the sorted path."""
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

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_observation(self, state):
        """
        Build a normalised 6-float observation vector.

        Each feature is scaled into roughly [-1, 1] / [0, 1] before being
        passed to StateNormalizer (which clips to [-5, 5]).  This gives
        VecNormalize's running-statistics a sensible starting range.

        Features:
          0  speed_norm           speed / 300          (0 → 1 at 300 km/h)
          1  dist_to_center_norm  dist / 50            (0 → 1 at 50 m off track)
          2  yaw_norm             yaw  / π             (-1 → 1)
          3  pitch_norm           pitch / π            (-1 → 1)
          4  roll_norm            roll  / π            (-1 → 1)
          5  checkpoints_norm     n_checkpoints / 20   (0 → 1 for most maps)
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
        Map continuous action array to TMInterface inputs.

        Steering is fully continuous: the float in [-1, 1] maps linearly to
        the integer range [-65536, 65536] that TM uses internally.
        Gas and brake are binary because TM's SetInputState only accepts 0/1
        for those inputs.
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
        # IMPORTANT: reward is accumulated across ALL messages that belong to
        # this single env step.  Previously the speed-reward line used `=`
        # (assignment) which wiped any checkpoint bonus collected earlier in
        # the same while-loop iteration.
        reward = 0.0

        while True:
            msgtype = self.iface._read_int32()

            if msgtype == int(MessageType.SC_RUN_STEP_SYNC):
                _time = self.iface._read_int32()
                state = self.iface.get_simulation_state()

                # -------------------------------------------------------
                # REWARD
                # -------------------------------------------------------

                # 1. Speed reward – linear in speed (not quadratic) so it
                #    doesn't completely dominate the progress signal.
                #    Reference: ~1.0 reward per step at 150 km/h.
                speed_factor = state.display_speed / 150.0
                reward += speed_factor * 1.0

                dist_to_center = 0.0
                if self.map_blocks:
                    pos = state.position
                    closest_idx, dist_to_center = self._closest_block_idx(pos)

                    # 2. Arc-length progress reward.
                    #    Award the net increase in metres along the path since
                    #    the episode start.  Scale: ~0.5 reward per 32 m block.
                    current_arc = self.arc_lengths[closest_idx]
                    if current_arc > self.highest_arc_length:
                        progress_m = current_arc - self.highest_arc_length
                        reward += progress_m * 0.015
                        self.highest_arc_length = current_arc

                    # 3. Gentle centerline nudge – does not overpower speed.
                    reward -= dist_to_center * 0.001

                    # 4. Per-step time penalty to prefer finishing quickly.
                    reward -= 0.005

                    # 5. Out-of-bounds: fell off elevated track.
                    if pos[1] < 20.0:
                        reward -= 10.0
                        terminated = True

                # 6. Crash: large sudden speed drop.
                speed_drop = self.previous_speed - state.display_speed
                if speed_drop > 50.0:
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

                # Save state for random-spawn curriculum.
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

                # Checkpoint bonus – accumulated into `reward` (not overwritten
                # later because the run-step branch now uses +=).
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
    # Reset
    # ------------------------------------------------------------------

    def _update_highest_arc_from_state(self, state):
        if not self.map_blocks:
            return
        idx, _ = self._closest_block_idx(state.position)
        self.highest_arc_length = self.arc_lengths[idx]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if not self.connected:
            self.connect()

        is_random_spawn = len(self.state_history) > 100 and random.random() < 0.5
        if is_random_spawn:
            self.iface.rewind_to_state(random.choice(self.state_history))
        else:
            self.iface.give_up()

        first_frame = True
        prev_time = -1

        while True:
            msgtype = self.iface._read_int32()
            if msgtype == int(MessageType.SC_RUN_STEP_SYNC):
                _time = self.iface._read_int32()
                state = self.iface.get_simulation_state()
                self.iface._respond_to_call(msgtype)

                if _time == 0:
                    obs = self._get_observation(state)
                    self.current_state = state
                    self.previous_speed = 0.0
                    self.consecutive_stuck_steps = 0
                    self.highest_arc_length = 0.0
                    break

                if is_random_spawn and first_frame:
                    obs = self._get_observation(state)
                    self.current_state = state
                    self.previous_speed = state.display_speed
                    self.consecutive_stuck_steps = 0
                    self._update_highest_arc_from_state(state)
                    break

                if first_frame:
                    prev_time = _time
                    first_frame = False
                    continue

                if _time > 0 and _time < prev_time - 100:
                    obs = self._get_observation(state)
                    self.current_state = state
                    self.previous_speed = 0.0
                    self.consecutive_stuck_steps = 0
                    self.highest_arc_length = 0.0
                    break

                prev_time = _time
            else:
                self.iface._respond_to_call(msgtype)

        return obs, {}

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------

    def close(self):
        if self.connected:
            self.iface.close()
            self.connected = False
