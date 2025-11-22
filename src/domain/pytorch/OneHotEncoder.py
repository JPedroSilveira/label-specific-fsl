from typing import List
from torch import Tensor
import torch

from config.type import DatasetConfig
from src.domain.device.DeviceGetter import DeviceGetter


class OneHotEncoder:
    @staticmethod
    def execute(tensor: Tensor, label_types: List[int], config: DatasetConfig) -> Tensor:
        num_classes = len(label_types)
        return torch.nn.functional.one_hot(tensor, num_classes).to(DeviceGetter.execute(), dtype=getattr(torch, config.data_type))