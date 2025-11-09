from torch import nn
from src.model.ClassifierModel import ClassifierModel
from src.selector.fs_based.mfs.v1.BaseMFSLayerV1Wrapper import BaseMFSLayerV1Wrapper
from src.selector.fs_based.mfs.BaseMFSSelectorWrapper import BaseMFSSelectorWrapper
from src.util.device_util import get_device

    
class MFSLayerV1ReLUSelector(BaseMFSSelectorWrapper):
    def __init__(self, n_features, n_labels):
        super().__init__(n_features, n_labels)
        device = get_device()
        self._activation = nn.ReLU()
        self._model = ClassifierModel(n_features, n_labels).to(device)
        self._model = BaseMFSLayerV1Wrapper(self._model, n_features, n_labels, self._activation).to(device)

    def get_name() -> str:
        return "Multiple FSL"