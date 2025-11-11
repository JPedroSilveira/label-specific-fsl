import numpy as np
from sklearn.linear_model import Lasso as SkLearnLasso

from src.domain.data.DatasetLoader import DatasetConfig
from src.domain.selector.types.enum.SelectorSpecificity import SelectorSpecificity
from src.domain.selector.types.base.BaseSelectorWeight import BaseSelectorWeight
from src.domain.data.types.Dataset import Dataset


class Lasso(BaseSelectorWeight):
    def __init__(self, n_features: int, n_labels: int, config: DatasetConfig) -> None:
        super().__init__(n_features, n_labels, config)
        self._model = SkLearnLasso(alpha=config.lasso_regularization, max_iter=10000)

    def get_name() -> str:
        return "Lasso"
    
    def can_predict(self) -> bool:
        return True
    
    def get_specificity(self) -> SelectorSpecificity:
        return SelectorSpecificity.PER_LABEL

    def fit(self, train_dataset: Dataset, _: Dataset) -> None:
        self._model.fit(train_dataset.get_features(), train_dataset.get_encoded_labels())
    
    def predict(self, dataset: Dataset) -> np.ndarray:
        y_pred = self.predict_probabilities(dataset)
        return np.argmax(y_pred, 1)
    
    def predict_probabilities(self, dataset: Dataset, use_softmax: bool=True) -> np.ndarray:
        return self._model.predict(dataset.get_features())
    
    def get_general_weights(self) -> np.ndarray:
        return np.max(self._model.coef_, axis=0)
    
    def get_per_label_weights(self) -> list[np.ndarray]:
        weights = []
        for class_weight in self._model.coef_.tolist():
            weights.append(np.array(class_weight))
        return weights