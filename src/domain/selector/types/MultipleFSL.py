import torch
import numpy as np
from torch import Tensor, nn

from config.type import DatasetConfig
from src.domain.device.DeviceGetter import DeviceGetter
from src.domain.pytorch.PyTorchPredict import PyTorchPredict
from src.domain.pytorch.PyTorchFit import PyTorchFit
from src.domain.data.types.Dataset import Dataset
from src.domain.model.ClassifierModel import ClassifierModel
from src.domain.selector.types.base.BaseSelectorWeight import BaseSelectorWeight
from src.domain.selector.types.base.BaseSelector import SelectorSpecificity

    
class MultipleFSL(BaseSelectorWeight):
    def __init__(self, n_features: int, n_labels: int, config: DatasetConfig) -> None:
        super().__init__(n_features, n_labels, config)
        self._model = ClassifierModel(n_features, n_labels, config).to(DeviceGetter.execute())
        self._model = MultipleFSLModel(self._model, n_features, n_labels, config.regularization_lambda).to(DeviceGetter.execute())

    def get_name() -> str:
        return "Multiple FSL"
    
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
    
    def get_per_label_weights(self) -> np.ndarray:
        w = self._model.get_activated_weight()
        result = []
        for i in range(0, self.get_n_labels()):
            result.append(w[i].clone().detach().cpu().numpy())
        return result
    
class MultipleFSLModel(nn.Module):
    def __init__(self, model: nn.Module, n_features: int, n_labels: int, regularization: float) -> None:
        super().__init__()
        self._weights = nn.Parameter(self.generate_initial_weights(n_features, n_labels).to(DeviceGetter.execute()))
        self._activation = nn.ReLU()
        self._model = model.to(DeviceGetter.execute())
        self._n_labels = n_labels
        self._n_features = n_features
        self._regularization = regularization
        self._label_selectors = Tensor(self.generate_label_selector(n_labels)).to(DeviceGetter.execute())

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
        #return self._regularization * torch.abs(torch.sum(self.get_weight()))
        #return self._regularization * torch.sum(self.get_activated_weight())
        #return torch.abs(self._n_features - (torch.sum(self.get_activated_weight()) / self._n_labels))

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
        return torch.ones(n_labels, n_features)