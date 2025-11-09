import torch
import numpy as np
from numpy import ndarray
from torch import nn, Tensor

from config.type import DatasetConfig
from src.device import device
from src.model.Dataset import Dataset
from src.model.ClassifierModel import ClassifierModel
from src.domain.pytorch.PyTorchPredict import PyTorchPredict
from src.domain.pytorch.PyTorchPerLabelFit import PyTorchPerLabelFit
from src.domain.selector.types.base.BaseSelectorWeight import BaseSelectorWeight
from src.domain.selector.types.base.BaseSelector import SelectorSpecificity


class ListenerFSL(BaseSelectorWeight):
    def __init__(self, n_features: int, n_labels: int, config: DatasetConfig) -> None:
        super().__init__(n_features, n_labels)
        self._model = ClassifierModel(n_features, n_labels).to(device)
        self._model = ListenerModel(self._model, n_features, n_labels, config.regularization_lambda).to(device)
        self._config = config

    def get_name() -> str:
        return "Listener"
    
    def can_predict(self) -> bool:
        return True
    
    def get_specificity(self) -> SelectorSpecificity:
        return SelectorSpecificity.PER_LABEL

    def fit(self, train_dataset: Dataset, test_dataset: Dataset) -> None: 
        PyTorchPerLabelFit.execute(self._model, train_dataset, self._config)

    def predict(self, dataset: Dataset) -> ndarray:
        y_pred = self.predict_probabilities(dataset)
        return np.argmax(y_pred, 1)
    
    def predict_probabilities(self, dataset: Dataset, use_softmax: bool=True) -> np.ndarray:
        return PyTorchPredict.execute(self._model, dataset.get_features(), use_softmax)
    
    def get_general_weights(self) -> ndarray:
        return self._model.get_activated_weight().clone().detach().cpu().numpy()[0]
    
    def get_weights_per_class(self) -> ndarray:
        return self._model.get_weight_per_class()
    
class ListenerModel(nn.Module):
    def __init__(self, model: nn.Module, n_features: int, n_labels: int, regularization: float) -> None:
        super().__init__()
        self._weight = nn.Parameter(torch.ones(1, n_features).to(device))
        self._activation = nn.ReLU()
        self._model = model
        self._n_labels = n_labels
        self._evaluators = torch.Tensor(self.generate_evaluators(n_features)).to(device)
        self._regularization = regularization

    def before_forward(self, y, _) -> None:
        unique_labels = torch.unique(y)
        if len(unique_labels) > 1:
            raise ValueError("Selector supports fit with only a single label")
        self._label = unique_labels[0]
        self._pre_weights = self.get_weight().detach().clone()

    def after_forward(self) -> None:
        new_weights = self.get_weight().detach()
        self._evaluators[self._label] = self._evaluators[self._label] + (new_weights - self._pre_weights)

    def forward(self, x: Tensor) -> Tensor:
        x = x * self.get_activated_weight()
        return self._model(x)

    def get_regularization(self) -> Tensor:
        return torch.sum(self._regularization * torch.sum(torch.abs(self.get_weight()), 1))
    
    def get_weight(self) -> Tensor:
        return self._weight
    
    def get_activated_weight(self) -> Tensor:
        return self._activation(self.get_weight())
    
    def get_weight_per_class(self) -> list[np.ndarray]:
        result = []
        for weights in self._evaluators:
            result.append(weights.clone().detach().cpu().numpy())
        return result
    
    def generate_evaluators(self, n_features: int) -> list:
        evaluators = []
        for _ in range(0, self._n_labels):
            evaluators.append(np.ones(n_features))
        return evaluators