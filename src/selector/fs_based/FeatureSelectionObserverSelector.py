import torch
import numpy as np
from typing import List
from src.config.general_config import REGULARIZATION_LAMBDA
from numpy import ndarray
from torch import nn, Tensor
from src.data.Dataset import Dataset
from src.selector.BaseWeightSelectorWrapper import BaseWeightSelectorWrapper
from src.selector.enum.PredictionMode import PredictionMode
from src.selector.enum.SelectionSpecificity import SelectionSpecificity
from src.model.ClassifierModel import ClassifierModel
from src.pytorch_helpers.PyTorchFit import pytorch_fit_by_label
from src.pytorch_helpers.PyTorchPredict import pytorch_predict_propabilities
from src.util.device_util import get_device


class FeatureSelectionLayer(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        device = get_device()
        self._fs_layer = nn.Linear(n_features, 1, bias=False).to(device)
        nn.init.constant_(self._fs_layer.weight, 1.0)
        self._activation = nn.ReLU()
        self._n_features = n_features

    def forward(self, x: Tensor):
        return x * self.get_activated_weight()

    def get_regularization(self) -> float:
        return REGULARIZATION_LAMBDA * torch.sum(torch.abs(self.get_weight()))
    
    def get_weight(self):
        return self._fs_layer.weight

    def get_activated_weight(self) -> Tensor:
        return self._activation(self.get_weight())

class FeatureSelectionObserverWrapper(nn.Module):
    def __init__(self, internal_model: nn.Module, n_features: int, n_labels: int):
        super().__init__()
        device = get_device()
        self._fs_layer = FeatureSelectionLayer(n_features).to(device)
        self._internal_model = internal_model
        self._n_labels = n_labels
        evaluators = []
        for _ in range(0, self._n_labels):
          evaluators.append(np.ones(n_features))
        self._evaluators = torch.Tensor(evaluators).to(device)

    def before_forward(self, y, _):
        unique_labels = torch.unique(y)
        if len(unique_labels) > 1:
            raise ValueError("Selector supports fit with only a single label")
        self._label = unique_labels[0]
        self._pre_weights = self.get_weight().detach().clone()

    def after_forward(self):
        new_weights = self.get_weight().detach()
        self._evaluators[self._label] = self._evaluators[self._label] + (new_weights - self._pre_weights)

    def forward(self, x: Tensor) -> Tensor:
        x = self._fs_layer(x)
        return self._internal_model(x)

    def get_regularization(self):
        return torch.sum(REGULARIZATION_LAMBDA * torch.sum(torch.abs(self.get_weight()), 1))

    def get_weight(self) -> torch.Tensor:
        return self._fs_layer.get_weight()
    
    def get_activated_weight(self) -> List:
        return self._fs_layer.get_activated_weight().clone().detach().cpu().numpy()
    
    def get_weight_per_class(self) -> List[List]:
        result = []
        for weights in self._evaluators:
            result.append(weights.clone().detach().cpu().numpy())
        return result

class FeatureSelectionObserverSelectorWrapper(BaseWeightSelectorWrapper):
    def __init__(self, n_features, n_labels):
        super().__init__(n_features, n_labels)
        device = get_device()
        self._model = ClassifierModel(n_features, n_labels).to(device)
        self._model = FeatureSelectionObserverWrapper(self._model, n_features, n_labels).to(device)

    def get_name() -> str:
        return "Listener"
    
    def get_prediction_mode(self) -> PredictionMode:
        return PredictionMode.AVAILABLE

    def get_selection_specificities(self):
        return [SelectionSpecificity.GENERAL, SelectionSpecificity.PER_LABEL]

    def fit(self, train_dataset: Dataset, test_dataset: Dataset): 
        pytorch_fit_by_label(self._model, train_dataset, test_dataset)

    def predict(self, dataset: Dataset) -> ndarray:
        y_pred = self.predict_probabilities(dataset)
        return np.argmax(y_pred, 1)
    
    def predict_probabilities(self, dataset: Dataset, use_softmax: bool=True) -> np.ndarray:
        return pytorch_predict_propabilities(self._model, dataset.get_features(), use_softmax)
    
    def get_general_weights(self) -> ndarray:
        return self._src.model.get_activated_weight()[0]
    
    def get_weights_per_class(self) -> ndarray:
        return self._src.model.get_weight_per_class()