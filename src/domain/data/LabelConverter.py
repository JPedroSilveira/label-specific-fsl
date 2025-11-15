from pandas import DataFrame

from config.type import DatasetConfig


class LabelConverter:
    def execute(dataframe: DataFrame, config: DatasetConfig) -> DataFrame:
        if config.label_type == "string":
            categorical_series = dataframe[config.label_column].astype('category')
            encoded_codes = categorical_series.cat.codes
            dataframe[config.label_column] = encoded_codes
            mapper = dict(enumerate(categorical_series.cat.categories))
            print("--- Conversion (Code -> Label) ---")
            print(mapper)
            mapper_reversed = {v: k for k, v in mapper.items()}
            print("--- Conversion (Label -> Code) ---")
            print(mapper_reversed)
            config.label_type = "long"
        
        