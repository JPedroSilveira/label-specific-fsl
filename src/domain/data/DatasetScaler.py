from sklearn import preprocessing

from config.type import DatasetConfig
from src.domain.data.types.SplittedDataset import SplittedDataset
from src.domain.data.types.Dataset import Dataset


class DatasetScaler:
    @staticmethod
    def execute(splitted_dataset: SplittedDataset, config: DatasetConfig) -> None:
        if config.normalize:
            DatasetScaler._scale(splitted_dataset.get_train())
            DatasetScaler._scale(splitted_dataset.get_test())
            
    @staticmethod
    def _scale(dataset: Dataset) -> None:
        scaler = preprocessing.RobustScaler().fit(dataset.get_features())
        dataset.set_features(scaler.transform(dataset.get_features()))