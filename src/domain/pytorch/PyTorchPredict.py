import torch
import numpy as np
from torch import nn

from src.util.numpy_util import convert_nparray_to_tensor


class PyTorchPredict:
    @staticmethod
    def execute(model: nn.Module, x: np.ndarray, use_softmax: bool=True) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            x_tensor = convert_nparray_to_tensor(x)
            y_pred = model(x_tensor)
            if use_softmax:
                return torch.nn.functional.softmax(y_pred, dim=1).cpu().numpy()
            else:
                return y_pred.cpu().numpy()