import sage
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.domain.selector.types.base.BaseSelectorWeight import BaseSelectorWeight
from src.domain.selector.types.enum.SelectorSpecificity import SelectorSpecificity
from src.domain.data.DatasetLoader import DatasetConfig
from src.domain.data.types.Dataset import Dataset


class SAGE(BaseSelectorWeight):
    def __init__(self, n_features: int, n_labels: int, config: DatasetConfig) -> None:
        super().__init__(n_features, n_labels, config)
        self._model = RandomForestClassifier(class_weight="balanced")
        self._k = config.sage_k
        self._representative_k = config.sage_representative_k

    def get_name() -> str:
        return "SAGE"
    
    def can_predict(self) -> bool:
        return True

    def get_specificity(self) -> SelectorSpecificity:
        return SelectorSpecificity.GENERAL

    def fit(self, train_dataset: Dataset, test_dataset: Dataset) -> None:
        self._model.fit(train_dataset.get_features(), train_dataset.get_labels())
        imputer = sage.MarginalImputer(
            self._model,
            test_dataset.get_features()[:self._representative_k]
        )
        estimator = sage.KernelEstimator(imputer, 'cross entropy')
        sage_values = estimator(test_dataset.get_features()[:self._k], test_dataset.get_labels()[:self._k])
        self._sage_values = sage_values.values

    def predict(self, dataset: Dataset) -> np.ndarray:
        return self._model.predict(dataset.get_features())
    
    def predict_probabilities(self, dataset: Dataset, use_softmax: bool=True) -> np.ndarray:
        return self._model.predict_proba(dataset.get_features())
    
    def get_general_weights(self) -> np.ndarray:
        return self._sage_values
