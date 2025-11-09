from torch import nn
from src.model.ClassifierModel import ClassifierModel
from src.selector.fs_based.fsr.v1.BaseFSRLayerV1Wrapper import BaseFSRLayerV1Wrapper
from src.selector.fs_based.fsr.BaseFSRSelectorWrapper import BaseFSRSelectorWrapper
from src.util.device_util import get_device

    
class FSRLayerV1TanhSelector(BaseFSRSelectorWrapper):
    def __init__(self, n_features, n_labels):
        super().__init__(n_features, n_labels)
        device = get_device()
        self._activation = nn.Tanh()
        self._model = ClassifierModel(n_features, n_labels).to(device)
        self._model = BaseFSRLayerV1Wrapper(self._model, n_features, n_labels, self._activation).to(device)

    def get_name() -> str:
        return "Output Aware - Tanh"