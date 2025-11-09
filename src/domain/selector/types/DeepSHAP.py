import shap
import numpy as np

from src.domain.pytorch.PyTorchPredict import PyTorchPredict
from src.domain.pytorch.PyTorchFit import PyTorchFit
from src.domain.selector.types.base.BaseSelectorWeight import BaseSelectorWeight
from src.model.SelectorSpecificity import SelectorSpecificity
from src.domain.data.DatasetLoader import DatasetConfig
from src.model.Dataset import Dataset
from src.model.ClassifierModel import ClassifierModel
from src.device import device

from src.util.array_util import get_weight_per_class_from_shap
from src.util.numpy_util import convert_nparray_to_tensor


class DeepSHAP(BaseSelectorWeight):
    def __init__(self, n_features: int, n_labels: int, config: DatasetConfig) -> None:
        super().__init__(n_features, n_labels)
        self._model = ClassifierModel(n_features, n_labels).to(device)
        self._k = config.shap_k
        self._representative_k = config.shap_representative_k
        self._config = config

    def get_name() -> str:
        return "DeepSHAP"
    
    def can_predict(self) -> bool:
        return False

    def get_specificity(self) -> SelectorSpecificity:
        return SelectorSpecificity.PER_LABEL

    def fit(self, train_dataset: Dataset, test_dataset: Dataset) -> None: 
        k = self._k
        if k > len(train_dataset.get_features()):
            k = len(train_dataset.get_features())
        representative_k = self._representative_k
        if representative_k > len(train_dataset.get_features()):
            representative_k = len(train_dataset.get_features())
        PyTorchFit.execute(self._model, train_dataset, self._config)
        shap_samples = shap.sample(test_dataset.get_features(), k)
        shap_representative = shap.sample(train_dataset.get_features(), representative_k)
        explainer = shap.DeepExplainer(model=self._model, data=convert_nparray_to_tensor(shap_representative))
        self._shap_values = explainer.shap_values(convert_nparray_to_tensor(shap_samples))

    def predict(self, dataset: Dataset) -> np.ndarray:
        y_pred = self.predict_probabilities(dataset)
        return np.argmax(y_pred, 1)
    
    def predict_probabilities(self, dataset: Dataset, use_softmax: bool=True) -> np.ndarray:
        return PyTorchPredict.execute(self._model, dataset.get_features(), use_softmax)
    
    def get_general_weights(self) -> np.ndarray:
        return np.max(self.get_weights_per_class(), axis=0)
    
    def get_weights_per_class(self) -> np.ndarray:
        return get_weight_per_class_from_shap(np.abs(self._shap_values).mean(0))