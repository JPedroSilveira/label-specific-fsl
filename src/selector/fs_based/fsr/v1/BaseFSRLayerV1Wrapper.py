from src.config.general_config import REGULARIZATION_LAMBDA
import torch
from random import random
from torch import nn
from src.selector.fs_based.fsr.BaseFSRLayer import BaseFSRLayer
from src.util.device_util import get_device
from src.util.torch_util import one_hot_encode


class BaseFSRLayerV1Wrapper(nn.Module):
    def __init__(self, internal_model, n_features, n_labels, activation):
        super().__init__()
        self._device = get_device()
        self._activation = activation
        self._fs_layer = BaseFSRLayer(n_features, n_labels, self._activation)
        self._internal_model = internal_model
        self._n_labels = n_labels
        self._n_features = n_features

    def before_forward(self, y, label_types):
        self._y = one_hot_encode(y, label_types)

    def forward(self, x):
        w = self._fs_layer.get_activated_weight()
        if self.training:
            x = x * torch.matmul(self._y, w)
        else:
            x = x * torch.sum(w, 0)
        return self._internal_model(x)
        
    def get_regularization(self):
        return REGULARIZATION_LAMBDA * torch.sum(torch.abs(self.get_weight()))

    def get_weight(self):
        return self._fs_layer.get_weight()
    
    def get_activated_weight(self):
        return self._fs_layer.get_activated_weight()