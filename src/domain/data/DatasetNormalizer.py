from sklearn import preprocessing

from config.type import DatasetConfig
from src.domain.data.types.SplittedDataset import SplittedDataset
from src.domain.data.types.Dataset import Dataset


class DatasetNormalizer:
    @classmethod
    def execute(cls, splitted_dataset: SplittedDataset, config: DatasetConfig) -> None:
        if config.normalize:
            cls._normalize(splitted_dataset.get_train())
            cls._normalize(splitted_dataset.get_test())
            
    @staticmethod
    def _normalize(dataset: Dataset) -> None:
        normalizer = preprocessing.StandardScaler().fit(dataset.get_features())
        dataset.set_features(normalizer.transform(dataset.get_features()))