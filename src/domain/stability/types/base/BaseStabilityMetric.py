import numpy as np
from pandas import DataFrame

from config.type import Config


class BaseStabilityMetric:
    @staticmethod
    def execute(dataframe1: DataFrame, dataframe2: DataFrame, config: Config) -> float:
        raise NotImplementedError()
    
    @staticmethod
    def get_name() -> str:
        raise NotImplementedError()