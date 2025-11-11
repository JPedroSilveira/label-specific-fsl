from sklearn import preprocessing

from config.type import DatasetConfig
from src.domain.data.types.SplittedDataset import SplittedDataset
from src.domain.data.types.Dataset import Dataset


class DatasetNormalizer:
    @staticmethod
    def execute(splitted_dataset: SplittedDataset, config: DatasetConfig) -> None:
        if config.normalize:
            DatasetNormalizer._normalize(splitted_dataset.get_train())
            DatasetNormalizer._normalize(splitted_dataset.get_test())
            
    @staticmethod
    def _normalize(dataset: Dataset) -> None:
        normalizer = preprocessing.StandardScaler().fit(dataset.get_features())
        dataset.set_features(normalizer.transform(dataset.get_features()))