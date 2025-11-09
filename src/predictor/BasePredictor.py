import numpy as np


class BasePredictor():
    def __init__(self, n_features, n_labels) -> None:
        pass

    def get_name() -> str:
        raise NotImplemented("")
    
    def get_class_name(self) -> str:
        return self.__class__.get_name()

    def should_encode_labels(self) -> bool:
        return False
     
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplemented("")

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplemented("")