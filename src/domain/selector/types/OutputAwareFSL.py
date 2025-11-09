import torch
import numpy as np
from torch import Tensor, nn

from config.type import DatasetConfig
from src.domain.pytorch.PyTorchFit import PyTorchFit
from src.domain.pytorch.PyTorchPredict import PyTorchPredict
from src.model.Dataset import Dataset
from src.device import device
from src.model.ClassifierModel import ClassifierModel
from src.domain.selector.types.base.BaseSelectorWeight import BaseSelectorWeight
from src.domain.selector.types.base.BaseSelector import SelectorSpecificity

from src.util.torch_util import one_hot_encode
    
class OutputAwareFSL(BaseSelectorWeight):
    def __init__(self, n_features: int, n_labels: int, config: DatasetConfig) -> None:
        super().__init__(n_features, n_labels)
        self._model = ClassifierModel(n_features, n_labels).to(device)
        self._model = OutputAwareFSLModel(self._model, n_features, n_labels, config.regularization_lambda).to(device)
        self._config = config

    def get_name() -> str:
        return "Output Aware"
    
    def can_predict(self) -> bool:
        return True
    
    def get_specificity(self) -> SelectorSpecificity:
        return SelectorSpecificity.PER_LABEL
    
    def fit(self, train_dataset: Dataset, test_dataset: Dataset) -> None: 
        PyTorchFit.execute(self._model, train_dataset, self._config)
    
    def predict(self, dataset: Dataset) -> np.ndarray:
        y_pred = self.predict_probabilities(dataset)
        return np.argmax(y_pred, 1)
    
    def predict_probabilities(self, dataset: Dataset, use_softmax: bool=True) -> np.ndarray:
        return PyTorchPredict.execute(self._model, dataset.get_features(), use_softmax)
    
    def get_general_weights(self) -> np.ndarray:
        return np.sum(self.get_weights_per_class(), axis=0)
    
    def get_weights_per_class(self) -> np.ndarray:
        w = self._src.model.get_activated_weight()
        result = []
        for i in range(0, self.get_n_labels()):
            result.append(w[i].clone().detach().cpu().numpy())
        return result
    
class OutputAwareFSLModel(nn.Module):
    def __init__(self, model: nn.Module, n_features: int, n_labels: int, regularization: float) -> None:
        super().__init__()
        self._weights = nn.Parameter(self.generate_initial_weights(n_features, n_labels).to(device))
        self._activation = nn.ReLU()
        self._model = model.to(device)
        self._n_labels = n_labels
        self._n_features = n_features
        self._regularization = regularization

    def forward(self, x) -> Tensor:
        w = self._activation(self._weights)
        if self.training:
            x = x * torch.matmul(self._y, w)
        else:
            x = x * torch.sum(w, 0)
        return self._model(x)
    
    def before_forward(self, y, label_types) -> None:
        self._y = one_hot_encode(y, label_types)

    def get_regularization(self) -> Tensor:
        return self._regularization * torch.sum(torch.abs(self.get_weight()))

    def get_weight(self) -> Tensor:
        return self._weights
    
    def get_activated_weight(self) -> Tensor:
        return self._activation(self.get_weight())
    
    def generate_initial_weights(self, n_features: int, n_labels: int) -> Tensor:
        return torch.ones(n_features, n_labels)