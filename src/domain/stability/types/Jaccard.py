from pandas import DataFrame

from config.type import Config
from src.domain.stability.types.base.BaseStabilityMetric import BaseStabilityMetric


class Jaccard(BaseStabilityMetric):
    @staticmethod
    def execute(dataframe1: DataFrame, dataframe2: DataFrame, config: Config) -> float:
        features1 = Jaccard._process_data(dataframe1, config)
        features2 = Jaccard._process_data(dataframe2, config)
        intersection = len(features1.intersection(features2))
        union = len(features1.union(features2))
        if union == 0:
            return 1.0 
        else:
            return intersection / union
    
    @staticmethod
    def get_name() -> str:
        return "Jaccard"
    
    @staticmethod
    def _process_data(dataframe: DataFrame, config: Config) -> str:
        top_k_dataframe = dataframe.nlargest(config.dataset.jaccard_k, 'value')
        return set(top_k_dataframe['feature'])