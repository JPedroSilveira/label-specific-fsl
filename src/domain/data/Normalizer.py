import numpy as np


class Normalizer:
    @staticmethod
    def execute(ndarray: np.ndarray) -> np.ndarray:
        # 1. Shift data so the minimum value is exactly 0
        shifted = ndarray - np.min(ndarray)
        # 2. Apply Log (adding 1 to avoid log(0))
        logged = np.log1p(shifted)
        # 3. Apply MinMax to fit [0, 1]
        min_val = np.min(logged)
        max_val = np.max(logged)
        # Avoid division by zero if all weights are identical
        if max_val - min_val == 0:
            return np.zeros_like(ndarray)
        return (logged - min_val) / (max_val - min_val)