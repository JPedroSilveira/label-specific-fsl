import torch
from torch.utils.data import Dataset
from src.util.device_util import get_device
from src.config.general_config import DATA_TYPE
  
def one_hot_encode(tensor, label_types):
  device = get_device()
  num_classes = len(label_types)
  return torch.nn.functional.one_hot(tensor, num_classes).to(device).to(DATA_TYPE)