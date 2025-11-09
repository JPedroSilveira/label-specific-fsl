import numpy as np
from src.config.general_config import PREDICTOR_EPOCHS
from src.predictor.BasePredictor import BasePredictor
from src.model.ClassifierModel import SimplerClassifierModel
from src.pytorch_helpers.PyTorchFit import pytorch_simple_fit
from src.pytorch_helpers.PyTorchPredict import pytorch_predict_propabilities
from src.util.device_util import get_device


class SequentialPredictor(BasePredictor):
    def __init__(self, n_features, n_labels) -> None:
        device = get_device()
        self._model = SimplerClassifierModel(n_features, n_labels).to(device)

    def get_name() -> str:
        return "Neural Network"

    def fit(self, X: np.ndarray, y: np.ndarray): 
        pytorch_simple_fit(self._model, X, y, n_epochs=PREDICTOR_EPOCHS)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        y_pred = pytorch_predict_propabilities(self._model, X)
        return np.argmax(y_pred, 1)