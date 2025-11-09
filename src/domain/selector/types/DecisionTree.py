import numpy as np
from typing import Any
from sklearn.tree import DecisionTreeClassifier

from config.type import DatasetConfig
from src.domain.selector.types.base.BaseSelectorWeight import BaseSelectorWeight
from src.model.SelectorSpecificity import SelectorSpecificity
from src.model.Dataset import Dataset


class DecisionTree(BaseSelectorWeight):
    def __init__(self, n_features: int, n_labels: int, config: DatasetConfig) -> None:
        super().__init__(n_features, n_labels)
        self.model = DecisionTreeClassifier()

    def get_name() -> str:
        return "Decision Tree"

    def can_predict(self) -> bool:
        return True
    
    def get_specificity(self) -> SelectorSpecificity:
        return SelectorSpecificity.GENERAL

    def fit(self, train_dataset: Dataset, _: Dataset) -> None: 
        self.src.model.fit(train_dataset.get_features(), train_dataset.get_labels())
    
    def predict(self, dataset: Dataset) -> Any:
        return self.src.model.predict(dataset.get_features())
    
    def predict_probabilities(self, dataset: Dataset, use_softmax: bool=True) -> np.ndarray:
        return self.src.model.predict_proba(dataset.get_features())
    
    def get_general_weights(self) -> np.ndarray:
        return self.src.model.feature_importances_