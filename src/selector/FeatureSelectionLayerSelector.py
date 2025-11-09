from src.config.general_config import REGULARIZATION_LAMBDA
import torch
import numpy as np
from torch import nn, Tensor
from src.model.Dataset import Dataset
from src.selector.BaseWeightSelectorWrapper import BaseWeightSelectorWrapper
from src.selector.enum.PredictionMode import PredictionMode
from src.selector.enum.SelectionSpecificity import SelectionSpecificity
from src.model.ClassifierModel import ClassifierModel
from src.pytorch_helpers.PyTorchFit import pytorch_fit
from src.pytorch_helpers.PyTorchPredict import pytorch_predict_propabilities
from src.util.device_util import get_device


class FeatureSelectionLayer(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        device = get_device()
        self._fs_layer = nn.Linear(n_features, 1, bias=False).to(device)
        nn.init.constant_(self._fs_layer.weight, 1.0)
        self._n_features = n_features

    def forward(self, x: Tensor):
        return x * self.get_weight()

    def get_regularization(self) -> float:
        return REGULARIZATION_LAMBDA * torch.abs(torch.sum(self.get_weight()) - 1)
    
    def get_weight(self):
        return self._fs_layer.weight

class FeatureSelectionLayerWrapper(nn.Module):
    def __init__(self, internal_model: nn.Module, n_features: int):
        super().__init__()
        device = get_device()
        self._fs_layer = FeatureSelectionLayer(n_features).to(device)
        self._internal_model = internal_src.model.to(device)

    def forward(self, x: Tensor) -> Tensor:
        x = self._fs_layer(x)
        return self._internal_model(x)

    def get_regularization(self):
        return self._fs_layer.get_regularization()

    def get_weight(self) -> torch.Tensor:
        return self._fs_layer.get_weight()

class FeatureSelectionLayerSelectorWrapper(BaseWeightSelectorWrapper):
    def __init__(self, n_features, n_labels):
        super().__init__(n_features, n_labels)
        device = get_device()
        self._model = ClassifierModel(n_features, n_labels).to(device)
        self._model = FeatureSelectionLayerWrapper(self._model, n_features).to(device)

    def get_name() -> str:
        return "FSL"

    def get_prediction_mode(self) -> PredictionMode:
        return PredictionMode.AVAILABLE

    def get_selection_specificities(self):
        return [SelectionSpecificity.GENERAL]

    def fit(self, train_dataset: Dataset, test_dataset: Dataset): 
        pytorch_fit(self._model, train_dataset)
    
    def predict(self, dataset: Dataset) -> np.ndarray:
        y_pred = self.predict_probabilities(dataset)
        return np.argmax(y_pred, 1)
    
    def predict_probabilities(self, dataset: Dataset, use_softmax: bool=True) -> np.ndarray:
        return pytorch_predict_propabilities(self._model, dataset.get_features(), use_softmax)
    
    def get_general_weights(self) -> np.ndarray:
        return self._src.model.get_weight()[0].detach().clone().cpu().numpy()