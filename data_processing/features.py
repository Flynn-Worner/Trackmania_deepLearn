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
        Clip already-scaled Trackmania telemetry/features to a stable range.

        `game_env.py` owns feature construction and scaling. Do not truncate
        here: the policy now receives Linesight-style lookahead path features
        in addition to the original basic telemetry.
        """
        return np.clip(np.array(state_array, dtype=np.float32), -5.0, 5.0)
