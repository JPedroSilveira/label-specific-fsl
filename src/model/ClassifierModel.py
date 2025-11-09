from src.config.general_config import SHOULD_USE_SIMPLER_MODEL
from torch import nn
from src.util.device_util import get_device


class SimplerClassifierModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimplerClassifierModel, self).__init__()
        device = get_device()
        self._model = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, output_size)
        ).to(device)

    def forward(self, x):
        return self._model(x)

class ClassifierModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(ClassifierModel, self).__init__()
        device = get_device()
        if SHOULD_USE_SIMPLER_MODEL:
            self._model = nn.Sequential(
                nn.Linear(input_size, 16),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(16, output_size)
            )
        else:
            self._model = nn.Sequential(
                nn.Linear(input_size, 100),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(100, 200),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(200, 100),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(100, 100),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(100, output_size)
            )
        self._model = self._src.model.to(device)

    def forward(self, x):
        return self._model(x)

class ClassifierModelWithSoftmax(nn.Module):
    def __init__(self, input_size, output_size):
        super(ClassifierModelWithSoftmax, self).__init__()
        device = get_device()
        self._model = nn.Sequential(
            ClassifierModel(input_size, output_size),
            nn.LogSoftmax(dim=1)
        ).to(device)

    def forward(self, x):
        return self._model(x)