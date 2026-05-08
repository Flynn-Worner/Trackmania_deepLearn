import numpy as np

class StateNormalizer:
    """
    Normalizes raw Trackmania telemetry data into a standardized [-1, 1] format.
    Neural networks (like PPO's Actor-Critic) learn much faster and more stably
    when inputs are scaled symmetrically around zero.
    """
    def __init__(self):
        pass

    def normalize(self, state_array):
        """
        Takes raw [speed, dist_to_center, yaw, pitch, roll, checkpoints] 
        The game_env.py already does division scaling. We just clip to avoid explosions.
        """
        # Ensure we always return exactly 6 floats
        return np.clip(np.array(state_array[:6], dtype=np.float32), -5.0, 5.0)
