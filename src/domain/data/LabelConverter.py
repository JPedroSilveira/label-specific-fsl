from pandas import DataFrame

from config.type import DatasetConfig
from src.domain.log.Logger import Logger


class LabelConverter:
    def execute(dataframe: DataFrame, config: DatasetConfig) -> DataFrame:
        if config.label_type == "string":
            categorical_series = dataframe[config.label_column].astype('category')
            encoded_codes = categorical_series.cat.codes
            dataframe[config.label_column] = encoded_codes
            mapper = dict(enumerate(categorical_series.cat.categories))
            Logger.execute("--- Conversion (Code -> Label) ---")
            Logger.execute(mapper)
            mapper_reversed = {v: k for k, v in mapper.items()}
            Logger.execute("--- Conversion (Label -> Code) ---")
            Logger.execute(mapper_reversed)
            config.label_type = "long"
        
        