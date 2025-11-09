import numpy as np
from scipy import stats
from src.evaluation.stability.metric.BaseWeightStabilityMetric import BaseWeightStabilityMetric


class PearsonMetric(BaseWeightStabilityMetric):
    def get_name():
        return "Pearson"
    
    def calculate_metric(self, selection1, selection2):
        result = stats.pearsonr(selection1, selection2).statistic
        return result