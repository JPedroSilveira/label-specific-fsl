import torch
import numpy as np
from src.util.device_util import get_device


def convert_nparray_to_tensor(x: np.ndarray, data_type: torch.dtype=torch.float32, device:str=None):
  device = get_device() if device is None else device
  return torch.tensor(x, dtype=data_type).to(device)

def get_interception_len(a1: np.ndarray, a2: np.ndarray):
    return len(np.intersect1d(a1, a2))

def normalize(nparray: np.ndarray):
    min_val = np.min(nparray)
    max_val = np.max(nparray)
    divider = max_val - min_val
    if divider == 0:
       return np.ones_like(nparray)
    return (nparray - min_val) / divider

def sort(nparray: np.ndarray):
   return np.sort(nparray)