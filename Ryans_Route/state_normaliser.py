"""Minimal observation normalizer for the TMNF environments."""

from __future__ import annotations

import numpy as np


class StateNormaliser:
    """Keep observations finite and within a stable range before VecNormalize."""

    def normalise(self, observation):
        obs = np.asarray(observation, dtype=np.float32)
        return np.clip(obs, -5.0, 5.0)

    # Backward-compatible US spelling used by older env code.
    def normalize(self, observation):
        return self.normalise(observation)


# Backward-compatible class name used by older imports.
class StateNormalizer(StateNormaliser):
    pass