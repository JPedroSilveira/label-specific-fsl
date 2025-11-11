from pandas import DataFrame
from scipy.stats import spearmanr

from config.type import Config
from src.domain.stability.types.base.BaseStabilityMetric import BaseStabilityMetric


class Spearman(BaseStabilityMetric):
    @staticmethod
    def execute(dataframe1: DataFrame, dataframe2: DataFrame, config: Config) -> float:
        s1 = Spearman._process_data(dataframe1)
        s2 = Spearman._process_data(dataframe2)
        all_features = s1.index.union(s2.index)
        weights1 = s1.reindex(all_features, fill_value=0)
        weights2 = s2.reindex(all_features, fill_value=0)
        spearman_corr, _ = spearmanr(weights1, weights2)
        return spearman_corr
    
    @staticmethod
    def get_name() -> str:
        return "Spearman"
    
    @staticmethod
    def _process_data(dataframe: DataFrame) -> str:
        return dataframe.set_index('feature')['value']