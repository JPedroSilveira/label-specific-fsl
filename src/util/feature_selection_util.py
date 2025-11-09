import numpy as np


def get_n_features_from_rank(feature_rank: np.ndarray, selection_size: int) -> list[int]:
    return feature_rank[:selection_size]

def remove_n_features_from_rank(feature_rank: np.ndarray, removal_size: int) -> list[int]:
    return feature_rank[removal_size:]

def remove_n_features_from_inversed_rank(feature_rank: np.ndarray, removal_size: int) -> list[int]:
    return feature_rank[::-1][removal_size:]