from typing import List
import numpy as np
import torch.utils.data as pydata
from src.config.general_config import BATCH_SIZE
from src.pytorch_helpers.PyTorchData import PyTorchData


def get_data_loader(X: np.ndarray, y: np.ndarray, device:str=None) -> pydata.DataLoader:
    pyTorchData = PyTorchData(X, y, device)
    return pydata.DataLoader(pyTorchData, shuffle=True, batch_size=BATCH_SIZE)

def get_by_label_data_loaders(X: np.ndarray, y: np.ndarray) -> List[pydata.DataLoader]:
    labels = np.unique(y)
    data_loader_by_label = {}
    for label_type in labels:
        label_X = []
        label_y = []
        for i, label in enumerate(y):
            if label_type == label:
                label_X.append(X[i].tolist())
                label_y.append(y[i].tolist())
        pyTorchData = PyTorchData(label_X, label_y)
        data_loader_by_label[label_type] = pydata.DataLoader(pyTorchData, shuffle=True, batch_size=BATCH_SIZE)
    return data_loader_by_label