import numpy as np
from typing import List

from src.domain.data.types.Dataset import Dataset


class SplittedDataset():
    def __init__(self, features_train: np.ndarray, features_test:np.ndarray, labels_train: np.ndarray, labels_test: np.ndarray, label_types: List, feature_names: List[str], 
                 informative_features: List[int], informative_features_per_label: dict[int, list[int]]) -> None:
        self._train_dataset = Dataset(
            features=features_train, 
            labels=labels_train,
            label_types=label_types, 
            feature_names=feature_names,
            informative_features=informative_features, 
            informative_features_per_label=informative_features_per_label
        )
        self._test_dataset = Dataset(
            features=features_test,
            labels=labels_test,
            label_types=label_types, 
            feature_names=feature_names,
            informative_features=informative_features, 
            informative_features_per_label=informative_features_per_label
        )
        self._labels_types = label_types
        self._informative_features = informative_features
        self._feature_names = feature_names
        self._informative_features_per_label = informative_features_per_label

    def get_train(self) -> Dataset:
        return self._train_dataset
    
    def get_test(self) -> Dataset:
        return self._test_dataset
    
    def get_n_features(self) -> int:
        return self._train_dataset.get_n_features()
    
    def get_n_labels(self) -> int:
        return self._test_dataset.get_n_labels()
    
    def get_label_types(self) -> list:
        return self._labels_types
        
    def get_feature_names(self) -> list[str]:
        return self._feature_names
    
    def get_informative_features(self) -> list[int]:
        return self._informative_features
    
    def get_informative_features_per_label(self) -> dict[int, list[int]]:
        return self._informative_features_per_label