import torch
import torch.utils.data as pydata
import numpy as np
from torch import Tensor

from config.type import DatasetConfig
from src.device import device
from src.config.general_config import DATA_TYPE, CLASS_TYPE
from src.util.numpy_util import convert_nparray_to_tensor


class PyTorchDataset(pydata.Dataset):
  def __init__(self, X: np.ndarray, y: np.ndarray, config: DatasetConfig) -> None:
    self._x = convert_nparray_to_tensor(X, data_type=getattr(torch, config.data_type), device=device)
    self._y = convert_nparray_to_tensor(y, data_type=getattr(torch, config.class_type), device=device)
    self._len = self._x.shape[0]

  def __getitem__(self, index) -> tuple[Tensor, Tensor]:
    return self._x[index], self._y[index]

  def __len__(self) -> int:
    return self._len
  
  def get_x(self) -> Tensor:
    return self._x
  
  def get_y(self) -> Tensor:
    return self._y