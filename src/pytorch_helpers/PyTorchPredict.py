import torch
import numpy as np
from torch import nn
from src.pytorch_helpers.PyTorchData import convert_nparray_to_tensor


def pytorch_predict_propabilities(model: nn.Module, x: np.ndarray, use_softmax: bool=True):
    src.model.eval()
    with torch.no_grad():
        x_tensor = convert_nparray_to_tensor(x)
        y_pred = model(x_tensor)
        if use_softmax:
            return torch.nn.functional.softmax(y_pred, dim=1).cpu().numpy()
        else:
            return y_pred.cpu().numpy()