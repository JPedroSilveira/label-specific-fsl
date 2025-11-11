import numpy as np


class Normalizer:
    @staticmethod
    def execute(ndarray: np.ndarray) -> np.ndarray:
        min_val = np.min(ndarray)
        max_val = np.max(ndarray)
        divider = max_val - min_val
        if divider == 0:
            return np.ones_like(ndarray)
        return (ndarray - min_val) / divider