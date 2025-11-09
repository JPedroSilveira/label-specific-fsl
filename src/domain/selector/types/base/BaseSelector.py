import numpy as np
from src.model.SelectorType import SelectorType
from src.model.SelectorSpecificity import SelectorSpecificity
from src.model.Dataset import Dataset


class BaseSelector():
    def __init__(self, n_features: int, n_labels :int) -> None:
        self._n_labels = n_labels
        self._n_features = n_features

    def get_name() -> str:
        raise NotImplemented("")

    def get_class_name(self) -> str:
        return self.__class__.get_name()
    
    def get_n_labels(self) -> int:
        return self._n_labels
    
    def get_n_features(self) -> int:
        return self._n_features

    def fit(self, train_dataset: Dataset, test_dataset: Dataset) -> None:
        raise NotImplemented("")
    
    def predict(self, dataset: Dataset) -> np.ndarray:
        raise NotImplemented("")
    
    def predict_probabilities(self, dataset: Dataset, use_softmax: bool=True) -> np.ndarray:
        raise NotImplemented("")
    
    def can_predict(self) -> bool:
        raise NotImplemented("")
    
    def get_specificity(self) -> SelectorSpecificity:
        raise NotImplemented("")
    
    def get_type(self) -> SelectorType:
        raise NotImplemented("")
    
    def get_general_weights(self) -> np.ndarray:
        raise NotImplemented("")
    
    def get_weights_per_class(self) -> np.ndarray[np.ndarray]:
        raise NotImplemented("")
    
    def get_general_ranking(self) -> np.ndarray:
        raise NotImplemented("")
    
    def get_ranking_per_class(self) -> list[np.ndarray]:
        raise NotImplemented("")