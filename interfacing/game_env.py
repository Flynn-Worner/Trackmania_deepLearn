import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .tminterface2 import TMInterface, MessageType

import sys
import os
# Ensure root is in path so we can import data_processing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_processing.features import StateNormalizer

class TrackmaniaEnv(gym.Env):
    """
    Custom Environment that follows gymnasium interface for Trackmania Nations Forever.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, port=8483, ticks_per_step=10):
        super(TrackmaniaEnv, self).__init__()
        
        self.port = port
        self.ticks_per_step = ticks_per_step # Number of engine ticks per RL action (10 = 10Hz)
        
        # Action space: Discrete actions for driving
        # 0: Do nothing, 1: Accelerate, 2: Brake, 3: Left, 4: Right, 5: Accel+Left, 6: Accel+Right
        self.action_space = spaces.Discrete(7)
        
        # State space: [speed, x, y, z, yaw, pitch, roll] 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        
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
        """Converts the raw TMInterface SimStateData into our observation space array."""
        speed = state.display_speed
        pos = state.position
        yaw, pitch, roll = state.yaw_pitch_roll
        raw_obs = np.array([speed, pos[0], pos[1], pos[2], yaw, pitch, roll], dtype=np.float32)
        
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
                
                # 1. Base Reward: Points for simply maintaining high speed
                reward = state.display_speed * 0.01 
                
                # 2. Crash Penalty: Detect massive speed loss (hitting a wall)
                speed_drop = self.previous_speed - state.display_speed
                if speed_drop > 50.0:  # Lost 50+ km/h in just 0.1 seconds
                    reward -= 5.0      # VERY SMALL penalty (don't make it terrified to drive)
                    terminated = True  # Terminate and force a reset
                    
                # 3. Stuck Penalty: Detect if the car is stopped/stuck for too long
                if state.display_speed < 10.0:
                    self.consecutive_stuck_steps += 1
                else:
                    self.consecutive_stuck_steps = 0
                    
                if self.consecutive_stuck_steps >= 50: # 50 steps @ 10Hz = 5 seconds
                    reward -= 50.0     # MASSIVE penalty for sitting still
                    terminated = True  # Terminate and force a reset
                
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
        
        # Wait for the game to process the reset, enter the countdown (-3000ms to 0ms), 
        # and then give us the very first frame of the actual race.
        seen_countdown = False
        while True:
            msgtype = self.iface._read_int32()
            if msgtype == int(MessageType.SC_RUN_STEP_SYNC):
                _time = self.iface._read_int32()
                state = self.iface.get_simulation_state()
                
                # ALWAYS respond to avoid deadlocking the game engine!
                self.iface._respond_to_call(msgtype)
                
                # If we see a negative time, we know the reset worked and we are in the countdown
                if _time < 0:
                    seen_countdown = True
                
                # Once we've seen the countdown, wait for time to hit 0 (the "GO!" signal)
                if seen_countdown and _time >= 0:
                    obs = self._get_observation(state)
                    self.current_state = state
                    self.previous_speed = 0.0
                    self.consecutive_stuck_steps = 0
                    break
            else:
                self.iface._respond_to_call(msgtype)

        info = {}
        return obs, info

    def close(self):
        if self.connected:
            self.iface.close()
            self.connected = False
