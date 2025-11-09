from typing import List
import numpy as np
import torch.utils.data as pydata

from config.type import DatasetConfig
from src.model.PyTorchDataset import PyTorchDataset

class PyTorchDataLoaderCreator:
    @staticmethod
    def execute(X: np.ndarray, y: np.ndarray, config: DatasetConfig) -> pydata.DataLoader:
        pyTorchData = PyTorchDataset(X, y, config)
        return pydata.DataLoader(pyTorchData, shuffle=True, batch_size=config.batch_size)