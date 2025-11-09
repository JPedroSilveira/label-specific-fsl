import torch
import inspect
import torch.utils.data as pydata
import numpy as np
from src.config.general_config import DATA_TYPE, CLASS_TYPE
from src.util.numpy_util import convert_nparray_to_tensor


class PyTorchData(pydata.Dataset):
  def __init__(self, X: np.ndarray, y: np.ndarray, device: str=None):
    self._x = convert_nparray_to_tensor(X, data_type=DATA_TYPE, device=device)
    self._y = convert_nparray_to_tensor(y, data_type=CLASS_TYPE, device=device)
    self._len = self._x.shape[0]

  def __getitem__(self, index):
    return self._x[index], self._y[index]

  def __len__(self):
    return self._len
  
  def get_x(self):
    return self._x
  
  def get_y(self):
    return self._y