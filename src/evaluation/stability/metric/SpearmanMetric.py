import numpy as np
from scipy import stats
from src.evaluation.stability.metric.BaseRankStabilityMetric import BaseRankStabilityMetric


class SpearmanMetric(BaseRankStabilityMetric):
    def get_name():
        return "Spearman"

    def calculate_metric(self, selection1, selection2):
        if np.array_equal(selection1, selection2):
            return 1.0
        return stats.spearmanr(selection1, selection2).statistic