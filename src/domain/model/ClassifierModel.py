from torch import Tensor, nn

from config.type import DatasetConfig
from src.domain.device.DeviceGetter import DeviceGetter


class ClassifierModel(nn.Module):
    def __init__(self, input_size: int, output_size: int, config: DatasetConfig) -> None:
        super(ClassifierModel, self).__init__()
        if config.model == 0:
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
        self._model = self._model.to(DeviceGetter.execute())

    def forward(self, x) -> Tensor:
        return self._model(x)