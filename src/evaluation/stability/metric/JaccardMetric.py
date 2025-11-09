import numpy as np
from scipy.spatial import distance
from src.evaluation.stability.metric.BaseRankStabilityMetric import BaseRankStabilityMetric


class JaccardMetric(BaseRankStabilityMetric):
    '''
    Jaccard is using the select ranking for calculation, but since it only considers the selected elements
    and not the order, it will evaluate only the selected subset
    '''
    def get_name():
        return "Jaccard"

    def calculate_metric(self, selection1: np.ndarray, selection2: np.ndarray):
        s1 = sorted(selection1)
        s2 = sorted(selection2)
        return 1 - distance.jaccard(s1, s2)