import torch
from torch import nn
from src.util.device_util import get_device


class BaseMFSLayer(nn.Module):
    def __init__(self, n_features, n_labels, activation):
        super().__init__()
        device = get_device()
        self._fs_layer = nn.Linear(n_features, n_labels, bias=False).to(device)
        nn.init.constant_(self._fs_layer.weight, 1.0)
        self._activation = activation

    def activation(self, x) -> torch.Tensor:
        return self._activation(x)

    def forward(self, x) -> torch.Tensor:
        raise NotImplementedError()

    def get_weight(self) -> torch.Tensor:
        return self._fs_layer.weight

    def get_activated_weight(self) -> torch.Tensor:
        return self.activation(self.get_weight())