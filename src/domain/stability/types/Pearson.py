from fractions import Fraction
import numpy as np
from pandas import DataFrame
from scipy.stats import pearsonr

from config.type import Config
from src.domain.stability.types.base.BaseStabilityMetric import BaseStabilityMetric


class Pearson(BaseStabilityMetric):
    @classmethod
    def execute(cls, dataframe1: DataFrame, dataframe2: DataFrame, config: Config) -> float:
        s1 = cls._process_data(dataframe1)
        s2 = cls._process_data(dataframe2)
        all_features = s1.index.union(s2.index)
        weights_a = s1.reindex(all_features, fill_value=0).to_numpy()
        weights_b = s2.reindex(all_features, fill_value=0).to_numpy()
        if np.array_equal(weights_a, weights_b):
            pearson_corr = 1.0
        else:
            pearson_corr, _ = pearsonr(weights_a, weights_b)
        return pearson_corr
    
    @staticmethod
    def get_name() -> str:
        return "Pearson"
    
    @staticmethod
    def _process_data(dataframe: DataFrame) -> str:
        return dataframe.set_index('feature')['value']