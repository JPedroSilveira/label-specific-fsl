from src.config.general_config import REGULARIZATION_LAMBDA
import torch
import numpy as np
from torch import nn, Tensor
from src.data.Dataset import Dataset
from src.model.ClassifierModel import ClassifierModel
from src.pytorch_helpers.PyTorchFit import pytorch_fit
from src.pytorch_helpers.PyTorchPredict import pytorch_predict_propabilities
from src.selector.BaseWeightSelectorWrapper import BaseWeightSelectorWrapper
from src.selector.enum.PredictionMode import PredictionMode
from src.selector.enum.SelectionMode import SelectionMode
from src.selector.enum.SelectionSpecificity import SelectionSpecificity
from src.util.device_util import get_device


class MFSLayer(nn.Module):
    def __init__(self, n_features, n_labels):
        super().__init__()
        device = get_device()
        self._fs_layer = nn.Linear(n_features, n_labels, bias=False).to(device)
        nn.init.constant_(self._fs_layer.weight, 1.0)
        self._relu = nn.ReLU()

    def activation(self, x) -> Tensor:
        return self._relu(x)
    
    def forward(self, x) -> torch.Tensor:
        w = self.get_activated_weight()
        return torch.matmul(x, w.T)
    
    def get_activated_weight(self) -> torch.Tensor:
        return self.activation(self.get_weight())

    def get_weight(self) -> torch.Tensor:
        return self._fs_layer.weight

class RFSLayerV1Wrapper(nn.Module):
    def __init__(self, internal_model, n_features, n_labels):
        super().__init__()
        device = get_device()
        self._fs_layer = MFSLayer(n_features, n_labels).to(device)
        self._internal_model = internal_src.model.to(device)

    def forward(self, x):
        general_weights = torch.sum(self._fs_layer.get_activated_weight(), 0)
        out_t = self._internal_model(x * general_weights)
        out_s = self._fs_layer(x)
        return out_t + out_s

    def get_regularization(self):
        return REGULARIZATION_LAMBDA * torch.sum(torch.abs(self.get_weight()))
    
    def get_weight(self):
        return self._fs_layer.get_weight()

    def get_activated_weight(self) -> torch.Tensor:
        return self._fs_layer.get_activated_weight()

class RFSLayerSelectorV1Wrapper(BaseWeightSelectorWrapper):
    def __init__(self, n_features, n_labels):
        super().__init__(n_features, n_labels)
        self._model = ClassifierModel(n_features, n_labels)
        self._model = RFSLayerV1Wrapper(self._model, n_features, n_labels)

    def get_name() -> str:
        return "Residual"

    def get_prediction_mode(self) -> PredictionMode:
        return PredictionMode.AVAILABLE

    def get_selection_specificities(self):
        return [SelectionSpecificity.PER_LABEL, SelectionSpecificity.GENERAL]

    def fit(self, train_dataset: Dataset, test_dataset: Dataset): 
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