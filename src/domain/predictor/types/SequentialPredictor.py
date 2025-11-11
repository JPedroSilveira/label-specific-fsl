import numpy as np

from config.type import DatasetConfig
from src.domain.device.DeviceGetter import DeviceGetter
from src.domain.predictor.types.base.BasePredictor import BasePredictor
from src.domain.pytorch.PyTorchPredict import PyTorchPredict
from src.domain.pytorch.PyTorchSimpleFit import PyTorchSimpleFit
from src.domain.model.ClassifierModel import ClassifierModel


class SequentialPredictor(BasePredictor):
    def __init__(self, n_features: int, n_labels: int, config: DatasetConfig) -> None:
        self._model = ClassifierModel(n_features, n_labels, config).to(DeviceGetter.execute())
        self._config = config

    def get_name() -> str:
        return "Neural Network"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None: 
        PyTorchSimpleFit.execute(self._model, X, y, self._config)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        y_pred = PyTorchPredict.execute(self._model, X)
        return np.argmax(y_pred, 1)