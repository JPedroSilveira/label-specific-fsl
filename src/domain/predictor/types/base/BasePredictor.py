import numpy as np

from config.type import DatasetConfig


class BasePredictor():
    def __init__(self, n_features: int, n_labels: int, config: DatasetConfig) -> None:
        pass

    def get_name() -> str:
        raise NotImplementedError()
    
    def get_class_name(self) -> str:
        return self.__class__.get_name()

    def should_encode_labels(self) -> bool:
        return False
     
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError()

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError()