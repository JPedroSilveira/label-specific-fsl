import numpy as np
from sklearn.svm import SVC

from config.type import DatasetConfig
from src.domain.predictor.types.base.BasePredictor import BasePredictor


class SVCPredictor(BasePredictor):
    def __init__(self, n_features: int, n_labels: int, config: DatasetConfig) -> None:
        self._model = SVC()

    def get_name() -> str:
        return "SVC"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None: 
        self._src.model.fit(X, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        y_pred = self._src.model.predict(X)
        return y_pred