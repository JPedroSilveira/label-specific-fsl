import numpy as np
from src.evaluation.stability.metric.BaseRankStabilityMetric import BaseRankStabilityMetric
from src.config.general_config import NUMBER_OF_FEATURES

class KunchevaMetric(BaseRankStabilityMetric):
    '''
    Kuncheva is using the select ranking for calculation, but since it only considers the selected elements
    and not the order, it will evaluate only the selected subset
    '''
    def get_name():
        return "Kuncheva"

    def calculate_metric(self, selection1: np.ndarray, selection2: np.ndarray):
        a = set(selection1)
        b = set(selection2)
        m = NUMBER_OF_FEATURES
        feat_size = len(a)
        same_size = feat_size == len(b)
        if m == feat_size:
            return 1.0

        if same_size and isinstance(a, set) and isinstance(b, set):
            r = a.intersection(b)
            k = feat_size
            return (len(r) * m - np.power(k, 2)) / (k * (m - k))
        else:
            raise TypeError("Only a pair of `sets` of the same size is allowed.")