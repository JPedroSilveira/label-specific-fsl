import torch
import numpy as np
from torch import nn, Tensor

from config.type import DatasetConfig
from src.domain.pytorch.PyTorchPredict import PyTorchPredict
from src.domain.pytorch.PyTorchFit import PyTorchFit
from src.domain.device.DeviceGetter import DeviceGetter
from src.domain.data.types.Dataset import Dataset
from src.domain.model.ClassifierModel import ClassifierModel
from src.domain.selector.types.enum.SelectorSpecificity import SelectorSpecificity
from src.domain.selector.types.base.BaseSelectorWeight import BaseSelectorWeight


class ResidualFSL(BaseSelectorWeight):
    def __init__(self, n_features: int, n_labels: int, config: DatasetConfig) -> None:
        super().__init__(n_features, n_labels, config)        
        self._model = ClassifierModel(n_features, n_labels, config).to(DeviceGetter.execute())
        self._model = ResidualModel(self._model, n_features, n_labels, config.regularization_lambda).to(DeviceGetter.execute())

    def get_name() -> str:
        return "Residual"
    
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
        return np.sum(self.get_per_label_weights(), axis=0)
    
    def get_per_label_weights(self) -> list[np.ndarray]:
        w = self._model.get_activated_weight()
        result = []
        for i in range(0, self.get_n_labels()):
            result.append(w[i].clone().detach().cpu().numpy())
        return result
    
class ResidualModel(nn.Module):
    def __init__(self, model, n_features: int, n_labels: int, regularization: float) -> None:
        super().__init__()
        self._weights = nn.Parameter(self.generate_initial_weights(n_features, n_labels).to(DeviceGetter.execute()))
        self._activation = nn.ReLU()
        self._model = model.to(DeviceGetter.execute())
        self._regularization = regularization

    def forward(self, x) -> Tensor:
        w = self.get_activated_weight()
        out_t = self._model(x * torch.sum(w, 0))
        out_s = torch.matmul(x, w.T)
        return out_t + out_s

    def get_regularization(self) -> Tensor:
        return self._regularization * torch.sum(torch.abs(self.get_weight()))
    
    def get_weight(self) -> Tensor:
        return self._weights
    
    def get_activated_weight(self) -> Tensor:
        return self._activation(self.get_weight())
    
    def generate_initial_weights(self, n_features: int, n_labels: int) -> Tensor:
        return torch.ones(n_labels, n_features)