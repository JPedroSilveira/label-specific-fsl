import numpy as np
from config.type import DatasetConfig
from lassonet import LassoNetClassifierCV 

from src.domain.device.DeviceGetter import DeviceGetter
from src.domain.selector.types.DecisionTree import BaseSelectorWeight, Dataset, SelectorSpecificity   


class LassoNet(BaseSelectorWeight):
    def __init__(self, n_features: int, n_labels: int, config: DatasetConfig) -> None:
        super().__init__(n_features, n_labels, config)
        self._model = LassoNetClassifierCV(
            hidden_dims=(100,200,100,100),
            dropout=0.1,
            device=DeviceGetter.execute(),
            random_state=config.random_seed,
            torch_seed=config.random_seed,
            verbose=True,
        )

    def get_name() -> str:
        return "LassoNet"
    
    def can_predict(self) -> bool:
        return True
    
    def get_specificity(self) -> SelectorSpecificity:
        return SelectorSpecificity.PER_LABEL
    
    def fit(self, train_dataset: Dataset, test_dataset: Dataset) -> None: 
        self._model.fit(train_dataset.get_features(), train_dataset.get_labels())
        self._per_class_importance = self._model.model.skip.weight.detach().cpu().numpy()

    def predict(self, dataset: Dataset) -> np.ndarray:
        return self._model.predict(dataset.get_features())
    
    def predict_probabilities(self, dataset: Dataset, use_softmax: bool=True) -> np.ndarray:
        return self._model.predict_proba(dataset.get_features())
    
    def get_general_weights(self) -> np.ndarray:
        return np.max(self.get_per_label_weights(), axis=0)
    
    def get_per_label_weights(self) -> np.ndarray:
        return np.abs(self._per_class_importance)