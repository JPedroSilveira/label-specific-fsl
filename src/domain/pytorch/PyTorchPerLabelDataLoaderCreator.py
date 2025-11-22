from typing import List
import numpy as np
import torch.utils.data as pydata

from config.type import DatasetConfig
from src.domain.data.types.PyTorchDataset import PyTorchDataset

class PyTorchPerLabelDataLoaderCreator:
    @staticmethod
    def execute(X: np.ndarray, y: np.ndarray, config: DatasetConfig) -> List[pydata.DataLoader]:
        labels = np.unique(y)
        data_loader_by_label = {}
        for label_type in labels:
            label_X = []
            label_y = []
            for i, label in enumerate(y):
                if label_type == label:
                    label_X.append(X[i].tolist())
                    label_y.append(y[i].tolist())
            pyTorchData = PyTorchDataset(label_X, label_y, config)
            data_loader_by_label[label_type] = pydata.DataLoader(pyTorchData, shuffle=False, batch_size=config.batch_size)
        return data_loader_by_label