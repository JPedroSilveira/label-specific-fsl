from os import path
import pandas as pd
from config.type import DatasetConfig


class DatasetLoader:
    @staticmethod
    def execute(config: DatasetConfig) -> pd.DataFrame:
        return pd.read_csv(path.join(config.root, config.filename))