from src.config.general_config import REGULARIZATION_LAMBDA
import torch
from torch import nn
from src.selector.fs_based.mfs.BaseMFSLayer import BaseMFSLayer
from src.util.device_util import get_device


class BaseMFSLayerV1Wrapper(nn.Module):
    def __init__(self, internal_model: nn.Module, n_features: int, n_labels: int, activation):
        super().__init__()
        self._device = get_device()
        self._activation = activation
        self._fs_layer = BaseMFSLayer(n_features, n_labels, self._activation).to(self._device)
        self._internal_model = internal_src.model.to(self._device)
        self._n_labels = n_labels
        self._n_features = n_features
        multipliers = []
        for i in range(0, n_labels):
          multiplier = []
          for u in range(0, n_labels):
            multiplier.append(1 if u == i else 0)
          multipliers.append(multiplier)
        self._multipliers = torch.Tensor(multipliers).to(self._device)

    def forward(self, x):
        weights_by_label = self.get_activated_weight()
        for i in range(0, self._n_labels):
            input = x * weights_by_label[i] 
            if i == 0:
                output = self._internal_model(input) * self._multipliers[i]
            else:
                output += self._internal_model(input) * self._multipliers[i]
        return output

    def get_regularization(self):
        return REGULARIZATION_LAMBDA * torch.sum(torch.abs(self.get_weight()))

    def get_weight(self):
        return self._fs_layer.get_weight()
    
    def get_activated_weight(self):
        return self._fs_layer.get_activated_weight()