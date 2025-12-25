import numpy as np
from pandas import DataFrame
from scipy.stats import spearmanr
from fractions import Fraction

from config.type import Config
from src.domain.stability.types.base.BaseStabilityMetric import BaseStabilityMetric


class Spearman(BaseStabilityMetric):
    @classmethod
    def execute(cls, dataframe1: DataFrame, dataframe2: DataFrame, config: Config) -> float:
        s1 = cls._process_data(dataframe1)
        s2 = cls._process_data(dataframe2)
        all_features = s1.index.union(s2.index)
        weights1 = s1.reindex(all_features, fill_value=0).to_numpy()
        weights2 = s2.reindex(all_features, fill_value=0).to_numpy()
        if np.array_equal(weights1, weights2):
            spearman_corr = 1.0
        else:
            spearman_corr, _ = spearmanr(weights1, weights2)
        return spearman_corr
    
    @staticmethod
    def get_name() -> str:
        return "Spearman"
    
    @staticmethod
    def _process_data(dataframe: DataFrame) -> str:
        return dataframe.set_index('feature')['value']