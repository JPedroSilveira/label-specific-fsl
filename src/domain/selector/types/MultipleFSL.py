import torch
import numpy as np
from torch import Tensor, nn

from config.type import DatasetConfig
from src.model.Dataset import Dataset
from src.device import device
from src.model.ClassifierModel import ClassifierModel
from src.domain.selector.types.base.BaseSelectorWeight import BaseSelectorWeight
from src.domain.selector.types.base.BaseSelector import SelectorSpecificity

from src.pytorch_helpers.PyTorchFit import pytorch_fit
from src.pytorch_helpers.PyTorchPredict import pytorch_predict_propabilities

    
class MultipleFSL(BaseSelectorWeight):
    def __init__(self, n_features: int, n_labels: int, config: DatasetConfig) -> None:
        super().__init__(n_features, n_labels)
        self._model = ClassifierModel(n_features, n_labels).to(device)
        self._model = MultipleFSLModel(self._model, n_features, n_labels, config.regularization_lambda).to(device)

    def get_name() -> str:
        return "Multiple FSL"
    
    def can_predict(self) -> bool:
        return True
    
    def get_specificity(self) -> SelectorSpecificity:
        return SelectorSpecificity.PER_LABEL
    
    def fit(self, train_dataset: Dataset, test_dataset: Dataset) -> None: 
        pytorch_fit(self._model, train_dataset)
    
    def predict(self, dataset: Dataset) -> np.ndarray:
        y_pred = self.predict_probabilities(dataset)
        return np.argmax(y_pred, 1)
    
    def predict_probabilities(self, dataset: Dataset, use_softmax: bool=True) -> np.ndarray:
        return pytorch_predict_propabilities(self._model, dataset.get_features(), use_softmax)
    
    def get_general_weights(self) -> np.ndarray:
        return np.sum(self.get_weights_per_class(), axis=0)
    
    def get_weights_per_class(self) -> np.ndarray:
        w = self._src.model.get_activated_weight()
        result = []
        for i in range(0, self.get_n_labels()):
            result.append(w[i].clone().detach().cpu().numpy())
        return result
    
class MultipleFSLModel(nn.Module):
    def __init__(self, model: nn.Module, n_features: int, n_labels: int, regularization: float) -> None:
        super().__init__()
        self._weights = nn.Parameter(self.generate_initial_weights(n_features, n_labels).to(device))
        self._activation = nn.ReLU()
        self._model = model.to(device)
        self._n_labels = n_labels
        self._n_features = n_features
        self._regularization = regularization
        self._label_selectors = Tensor(self.generate_label_selector(n_labels)).to(device)

    def forward(self, x) -> Tensor:
        weights_by_label = self.get_activated_weight()
        for i in range(0, self._n_labels):
            input = x * weights_by_label[i] 
            if i == 0:
                output = self._model(input) * self._label_selectors[i]
            else:
                output += self._model(input) * self._label_selectors[i]
        return output

    def get_regularization(self) -> Tensor:
        return self._regularization * torch.sum(torch.abs(self.get_weight()))

    def get_weight(self) -> Tensor:
        return self._weights
    
    def get_activated_weight(self) -> Tensor:
        return self._activation(self.get_weight())
    
    def generate_label_selector(self, n_labels: int) -> list[list[int]]:
        label_selectors = []
        for i in range(0, n_labels):
            label_selector = []
            for u in range(0, n_labels):
                label_selector.append(1 if u == i else 0)
            label_selectors.append(label_selector)
        return label_selectors
    
    def generate_initial_weights(self, n_features: int, n_labels: int) -> Tensor:
        return torch.ones(n_features, n_labels)