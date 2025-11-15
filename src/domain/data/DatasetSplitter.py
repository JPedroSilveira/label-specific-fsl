from sklearn.model_selection import train_test_split

from config.type import DatasetConfig
from src.domain.data.types.SplittedDataset import SplittedDataset
from src.domain.data.types.Dataset import Dataset


class DatasetSplitter:
    @staticmethod
    def execute(dataset: Dataset, test_size: int, config: DatasetConfig) -> SplittedDataset:
        label_types = dataset.get_label_types()
        feature_names = dataset.get_feature_names()
        informative_features = dataset.get_informative_features()
        informative_features_per_label = dataset.get_informative_features_per_label()
        features = dataset.get_features()
        labels = dataset.get_labels()
        features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=test_size, stratify=labels, shuffle=True, random_state=config.random_seed)
        return SplittedDataset(
            features=features,
            labels=labels,
            features_train=features_train, 
            features_test=features_test,
            labels_train=labels_train,
            labels_test=labels_test,
            label_types=label_types, 
            feature_names=feature_names,
            informative_features=informative_features, 
            informative_features_per_label=informative_features_per_label
        )