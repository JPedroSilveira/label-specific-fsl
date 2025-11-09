import numpy as np
from typing import Any
from sklearn.ensemble import RandomForestClassifier

from src.model.SelectorSpecificity import SelectorSpecificity
from src.domain.selector.types.base.BaseSelectorWeight import BaseSelectorWeight
from src.model.Dataset import Dataset


class RandomForest(BaseSelectorWeight):
    def __init__(self, n_features, n_labels) -> None:
        super().__init__(n_features, n_labels)
        self._model = RandomForestClassifier()

    def get_name() -> str:
        return "Random Forest"
    
    def can_predict(self) -> bool:
        return True

    def get_specificity(self) -> SelectorSpecificity:
        return SelectorSpecificity.GENERAL

    def fit(self, train_dataset: Dataset, test_dataset: Dataset) -> None:
        self._src.model.fit(train_dataset.get_features(), train_dataset.get_labels())
    
    def predict(self, dataset: Dataset) -> Any:
        return self._src.model.predict(dataset.get_features())
    
    def predict_probabilities(self, dataset: Dataset, use_softmax: bool=True) -> np.ndarray:
        return self._src.model.predict_proba(dataset.get_features())
    
    def get_general_weights(self) -> np.ndarray:
        return self._src.model.feature_importances_