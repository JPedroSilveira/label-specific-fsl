from pandas import DataFrame
from scipy.stats import pearsonr

from config.type import Config
from src.domain.stability.types.base.BaseStabilityMetric import BaseStabilityMetric


class Pearson(BaseStabilityMetric):
    @staticmethod
    def execute(dataframe1: DataFrame, dataframe2: DataFrame, config: Config) -> float:
        s1 = Pearson._process_data(dataframe1)
        s2 = Pearson._process_data(dataframe2)
        all_features = s1.index.union(s2.index)
        weights_a = s1.reindex(all_features, fill_value=0)
        weights_b = s2.reindex(all_features, fill_value=0)
        pearson_corr, _ = pearsonr(weights_a, weights_b)
        return pearson_corr
    
    @staticmethod
    def get_name() -> str:
        return "Pearson"
    
    @staticmethod
    def _process_data(dataframe: DataFrame) -> str:
        return dataframe.set_index('feature')['value']