import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .tminterface2 import TMInterface, MessageType

import sys
import os
import json
import math
# Ensure root is in path so we can import data_processing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_processing.features import StateNormalizer

class TrackmaniaEnv(gym.Env):
    """
    Custom Environment that follows gymnasium interface for Trackmania Nations Forever.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, port=8483, ticks_per_step=25):
        super(TrackmaniaEnv, self).__init__()
        
        self.port = port
        self.ticks_per_step = ticks_per_step # Number of engine ticks per RL action (10 = 10Hz)
        
        # Action space: Discrete actions for driving
        # 0: Do nothing, 1: Accelerate, 2: Brake, 3: Left, 4: Right, 5: Accel+Left, 6: Accel+Right
        self.action_space = spaces.Discrete(7)
        
        # We removed Absolute GPS Coordinates (X,Y,Z).
        # We added Distance to Centerline.
        # Observation is now 6 floats: [speed, dist_to_center, yaw, pitch, roll, checkpoints_hit]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        
        # Load the extracted map blocks for the centerline!
        self.map_blocks = []
        blocks_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "map_blocks.json")
        if os.path.exists(blocks_path):
            with open(blocks_path, "r") as f:
                raw_blocks = json.load(f)
            
            # Sort the blocks into a topological "Snake" path starting from the Start Line!
            start_idx = 0
            for i, b in enumerate(raw_blocks):
                if 'start' in b['name'].lower():
                    start_idx = i
                    break
                    
            if raw_blocks:
                self.map_blocks = [raw_blocks[start_idx]]
                unvisited = raw_blocks.copy()
                unvisited.pop(start_idx)
                
                # Greedily connect the closest blocks to form the path
                while unvisited:
                    last_block = self.map_blocks[-1]["world_center"]
                    closest_idx = 0
                    min_d = float('inf')
                    for i, b in enumerate(unvisited):
                        center = b["world_center"]
                        # We use 2D distance for sorting the track path to ignore vertical bumps
                        d = math.sqrt((last_block["x"] - center["x"])**2 + (last_block["z"] - center["z"])**2)
                        if d < min_d:
                            min_d = d
                            closest_idx = i
                    self.map_blocks.append(unvisited.pop(closest_idx))
                    
            print(f"Loaded and path-sorted {len(self.map_blocks)} map blocks for Spline Progress Tracking!")
        else:
            print("WARNING: map_blocks.json not found! Centerline tracking will not work.")
            
        self.highest_block_idx = 0
        
        self.iface = TMInterface(self.port)
        self.connected = False
        
        self.current_state = None
        self.last_position = None
        self.total_reward = 0
        self.previous_speed = 0.0
        self.consecutive_stuck_steps = 0
        
        self.normalizer = StateNormalizer()
        
        # To handle the TMInterface message loop
        self.pending_action = None
        self.step_completed = False

    def connect(self):
        if not self.connected:
            print(f"Connecting to TMInterface on port {self.port}...")
            self.iface.register(timeout=None)
            self.connected = True
            
            # Setup TMInterface properties
            # Wait for initial connection sync message
            while True:
                msgtype = self.iface._read_int32()
                if msgtype == int(MessageType.SC_ON_CONNECT_SYNC):
                    # Request updates every X ticks
                    self.iface.set_on_step_period(self.ticks_per_step)
                    self.iface._respond_to_call(msgtype)
                    break
                else:
                    self.iface._respond_to_call(msgtype)
                    
            print("Successfully connected and synced to the game environment!")

    def _get_observation(self, state):
        """Converts the raw TMInterface SimStateData into our generalized observation space array."""
        speed = state.display_speed
        pos = state.position
        yaw, pitch, roll = state.yaw_pitch_roll
        
        checkpoints = state.cp_data.cp_times_length if state.cp_data else 0
        
        # Find closest block distance
        min_dist = 0.0
        if self.map_blocks:
            min_dist = float('inf')
            for b in self.map_blocks:
                center = b["world_center"]
                dist = math.sqrt((pos[0] - center["x"])**2 + (pos[2] - center["z"])**2)
                if dist < min_dist:
                    min_dist = dist
                    
        # Notice how X, Y, Z coordinates are completely GONE!
        # The bot must learn using only speed, distance to center, and angles!
        raw_obs = np.array([
            float(speed), 
            float(min_dist), 
            float(yaw), 
            float(pitch), 
            float(roll),
            float(checkpoints)
        ], dtype=np.float32)
        
        # Pass through the normalizer before giving it to the neural network
        return self.normalizer.normalize(raw_obs)

    def _apply_action(self, action):
        """Maps our discrete action integer to trackmania inputs."""
        # Defaults
        left, right, accelerate, brake = False, False, False, False
        
        if action == 1: accelerate = True
        elif action == 2: brake = True
        elif action == 3: left = True
        elif action == 4: right = True
        elif action == 5: accelerate = True; left = True
        elif action == 6: accelerate = True; right = True
            
        self.iface.set_input_state(left=left, right=right, accelerate=accelerate, brake=brake)

    def step(self, action):
        if not self.connected:
            self.connect()

        # Send our action to the game
        self._apply_action(action)
        
        # Wait for the game to simulate physics for `ticks_per_step` 
        # and give us the next state sync
        terminated = False
        truncated = False
        reward = 0.0
        
        while True:
            msgtype = self.iface._read_int32()
            
            if msgtype == int(MessageType.SC_RUN_STEP_SYNC):
                _time = self.iface._read_int32()
                state = self.iface.get_simulation_state()
                
                # We reached our physics tick, calculate observation
                obs = self._get_observation(state)
                
                # ==========================================
                # REWARD SYSTEM
                # ==========================================
                
                # 1. Base Reward: Speed is KING! We must massively reward moving!
                # 100 km/h = +2.0 reward per frame.
                reward = state.display_speed * 0.02 
                
                dist_to_center = 0.0
                if self.map_blocks:
                    pos = state.position
                    min_dist = float('inf')
                    closest_idx = 0
                    
                    for i, b in enumerate(self.map_blocks):
                        center = b["world_center"]
                        # We use 3D distance to find the absolute closest block
                        dist = math.sqrt((pos[0] - center["x"])**2 + (pos[1] - center["y"])**2 + (pos[2] - center["z"])**2)
                        if dist < min_dist:
                            min_dist = dist
                            closest_idx = i
                    
                    closest_center = self.map_blocks[closest_idx]["world_center"]
                    
                    # We use 3D distance to cleanly catch falling off elevated tracks 
                    # without falsely triggering on downhill slopes or undulating terrain!
                    dist_to_center = math.sqrt((pos[0] - closest_center["x"])**2 + (pos[1] - closest_center["y"])**2 + (pos[2] - closest_center["z"])**2)
                    
                    # 2. Forward Progress Reward! (The Snake Logic)
                    # We reward the AI specifically for pushing further down the path!
                    if closest_idx > self.highest_block_idx:
                        progress_blocks = closest_idx - self.highest_block_idx
                        reward += progress_blocks * 5.0  # Massive +5.0 reward for breaking new ground!
                        self.highest_block_idx = closest_idx
                        
                    # 3. Continuous Spline Penalty
                    # We make this penalty extremely tiny compared to the speed reward.
                    # We just want to gently 'nudge' the car, not terrify it into holding the brake.
                    # e.g., 10m away = -0.05 penalty.
                    reward -= dist_to_center * 0.005
                    
                    # 4. Marginal Step Penalty (Time Penalty)
                    # This gently bleeds points to encourage the AI to finish the race quickly 
                    # rather than driving in circles to farm speed points.
                    reward -= 0.1
                    
                    # 5. Out of Bounds Detection!
                    # Trackmania blocks are 32m wide. The absolute farthest corner is 22.6m away. 
                    # If 3D distance > 24m, the car has completely fallen off the snake track!
                    if dist_to_center > 24.0:
                        reward -= 50.0
                        terminated = True
                        print(f"Fell off! 3D Dist to center: {dist_to_center:.1f}m")
                
                # 3. Crash Penalty: Detect massive speed loss (hitting a wall)
                speed_drop = self.previous_speed - state.display_speed
                if speed_drop > 50.0:  # Lost 50+ km/h in just 0.1 seconds
                    reward -= 50.0     # Heavy penalty
                    terminated = True  # Terminate and force a reset
                    
                # 3. Stuck Penalty: Detect if the car is stopped/stuck for too long
                if state.display_speed < 10.0:
                    self.consecutive_stuck_steps += 1
                else:
                    self.consecutive_stuck_steps = 0
                    
                if self.consecutive_stuck_steps >= 50: # 50 steps @ 10Hz = 5 seconds
                    reward -= 50.0     # MASSIVE penalty for sitting still
                    terminated = True  # Terminate and force a reset
                    # print("Stuck penalty triggered!")
                
                self.previous_speed = state.display_speed
                # ==========================================
                
                self.current_state = state
                self.iface._respond_to_call(msgtype)
                break # Exit loop, step is complete
                
            elif msgtype == int(MessageType.SC_CHECKPOINT_COUNT_CHANGED_SYNC):
                current = self.iface._read_int32()
                target = self.iface._read_int32()
                
                # Progressive Reward: massive points for clearing a checkpoint!
                reward += 50.0
                
                # If we hit the final finish line
                if current == target:
                    terminated = True
                    reward += 100.0 # Extra bonus for finishing the race
                    print(f"FINISH LINE REACHED!!!")
                    
                self.iface._respond_to_call(msgtype)
                
            else:
                # Handle any other async messages
                self.iface._respond_to_call(msgtype)
                
        info = {}
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if not self.connected:
            self.connect()
            
        # Give up to reset the car to the starting line
        self.iface.give_up()
        
        # Wait for the game to process the reset.
        first_frame = True
        prev_time = -1
        while True:
            msgtype = self.iface._read_int32()
            if msgtype == int(MessageType.SC_RUN_STEP_SYNC):
                _time = self.iface._read_int32()
                state = self.iface.get_simulation_state()
                
                # ALWAYS respond to avoid deadlocking the game engine!
                self.iface._respond_to_call(msgtype)
                
                # If time is exactly 0, the race has officially started!
                if _time == 0:
                    obs = self._get_observation(state)
                    self.current_state = state
                    self.previous_speed = 0.0
                    self.consecutive_stuck_steps = 0
                    self.highest_block_idx = 0
                    break
                
                # On the very first frame, we just record the time and wait.
                if first_frame:
                    prev_time = _time
                    first_frame = False
                    continue
                
                # If time drops massively (e.g. 5000 down to 100), the reset happened!
                # This catches maps with no countdown where time never hits exactly 0.
                if _time > 0 and _time < prev_time - 100:
                    obs = self._get_observation(state)
                    self.current_state = state
                    self.previous_speed = 0.0
                    self.consecutive_stuck_steps = 0
                    self.highest_block_idx = 0
                    break
                    
                prev_time = _time
            else:
                self.iface._respond_to_call(msgtype)

        info = {}
        return obs, info

    def close(self):
        if self.connected:
            self.iface.close()
            self.connected = False
