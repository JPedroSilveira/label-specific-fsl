from typing import List
import numpy as np
from config.type import DatasetConfig
from src.domain.selector.types.enum.SelectorType import SelectorType
from src.domain.selector.types.enum.SelectorSpecificity import SelectorSpecificity
from src.domain.data.types.Dataset import Dataset


class BaseSelector():
    def __init__(self, n_features: int, n_labels :int, config: DatasetConfig) -> None:
        self._n_labels = n_labels
        self._n_features = n_features
        self._config = config

    def get_name() -> str:
        raise NotImplementedError()

    def get_selector_name(self) -> str:
        return self.__class__.get_name()
    
    def get_n_labels(self) -> int:
        return self._n_labels
    
    def get_n_features(self) -> int:
        return self._n_features

    def fit(self, train_dataset: Dataset, test_dataset: Dataset) -> None:
        raise NotImplementedError()
    
    def predict(self, dataset: Dataset) -> np.ndarray:
        raise NotImplementedError()
    
    def predict_probabilities(self, dataset: Dataset, use_softmax: bool=True) -> np.ndarray:
        raise NotImplementedError()
    
    def can_predict(self) -> bool:
        raise NotImplementedError()
    
    def get_specificity(self) -> SelectorSpecificity:
        raise NotImplementedError()
    
    def get_type(self) -> SelectorType:
        raise NotImplementedError()
    
    def get_general_weights(self) -> np.ndarray:
        raise NotImplementedError()
    
    def get_per_label_weights(self) -> List[np.ndarray]:
        raise NotImplementedError()
    
    def get_general_ranking(self) -> np.ndarray:
        raise NotImplementedError()
    
    def get_per_label_ranking(self) -> List[np.ndarray]:
        raise NotImplementedError()