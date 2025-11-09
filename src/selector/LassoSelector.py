import numpy as np
from sklearn.linear_model import Lasso
from src.data.Dataset import Dataset
from src.selector.BaseWeightSelectorWrapper import BaseWeightSelectorWrapper
from src.selector.enum.SelectionMode import SelectionMode
from src.selector.enum.PredictionMode import PredictionMode
from src.selector.enum.SelectionSpecificity import SelectionSpecificity


class LassoSelectorWrapper(BaseWeightSelectorWrapper):
    def __init__(self, n_features: int, n_labels):
        super().__init__(n_features, n_labels)
        regularization_strength = 0.01
        self._model = Lasso(alpha=regularization_strength, max_iter=10000)

    def get_name() -> str:
        return "Lasso"

    def fit(self, train_dataset: Dataset, _: Dataset):
        self._src.model.fit(train_dataset.get_features(), train_dataset.get_encoded_labels())
    
    def predict(self, dataset: Dataset):
        y_pred = self.predict_probabilities(dataset)
        return np.argmax(y_pred, 1)
    
    def predict_probabilities(self, dataset: Dataset, use_softmax: bool=True) -> np.ndarray:
        return self._src.model.predict(dataset.get_features())

    def get_prediction_mode(self) -> PredictionMode:
        return PredictionMode.AVAILABLE
    
    def get_selection_specificities(self):
        return [SelectionSpecificity.PER_LABEL, SelectionSpecificity.GENERAL]
    
    def get_general_weights(self) -> np.ndarray:
        return np.max(self._src.model.coef_, axis=0)
    
    def get_weights_per_class(self) -> list[np.ndarray]:
        weights = []
        for class_weight in self._src.model.coef_.tolist():
            weights.append(np.array(class_weight))
        return weights