import torch
import numpy as np
from torch import nn

from src.util.numpy_util import convert_nparray_to_tensor


class PyTorchPredict:
    @staticmethod
    def execute(model: nn.Module, x: np.ndarray, use_softmax: bool=True, return_tensor: bool=False) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            if type(x) is not torch.Tensor:
                x = convert_nparray_to_tensor(x)
            y_pred = model(x)
            if use_softmax:
                y_pred = torch.nn.functional.softmax(y_pred, dim=1)
            if not return_tensor:
                y_pred = y_pred.cpu().numpy()
        return y_pred