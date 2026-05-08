import numpy as np

class StateNormalizer:
    """
    Normalizes raw Trackmania telemetry data into a standardized [-1, 1] format.
    Neural networks (like PPO's Actor-Critic) learn much faster and more stably
    when inputs are scaled symmetrically around zero.
    """
    def __init__(self):
        # Estimated maximum values for normalization (Stadium car)
        self.max_speed = 400.0  # max speed km/h
        
        # Trackmania Nations map boundaries can vary, but these are safe bounds
        # X and Z are horizontal, Y is vertical.
        self.max_coord = 1000.0
        self.max_y = 200.0
        
        # Angles in radians are bounded by [-pi, pi] natively
        self.max_angle = np.pi

    def normalize(self, state_array):
        """
        Takes raw [speed, x, y, z, yaw, pitch, roll] and returns normalized values.
        """
        speed = np.clip(state_array[0] / self.max_speed, -1.0, 1.0)
        
        # Map coordinates
        x = np.clip(state_array[1] / self.max_coord, -1.0, 1.0)
        y = np.clip(state_array[2] / self.max_y, -1.0, 1.0)
        z = np.clip(state_array[3] / self.max_coord, -1.0, 1.0)
        
        # Angles
        yaw = state_array[4] / self.max_angle
        pitch = state_array[5] / self.max_angle
        roll = state_array[6] / self.max_angle
        
        return np.array([speed, x, y, z, yaw, pitch, roll], dtype=np.float32)
