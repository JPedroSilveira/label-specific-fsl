import shap
import numpy as np

from src.domain.device.DeviceGetter import DeviceGetter
from src.domain.pytorch.PyTorchPredict import PyTorchPredict
from src.domain.pytorch.PyTorchFit import PyTorchFit
from src.domain.selector.types.base.BaseSelectorWeight import BaseSelectorWeight
from src.domain.selector.types.enum.SelectorSpecificity import SelectorSpecificity
from src.domain.data.DatasetLoader import DatasetConfig
from src.domain.data.types.Dataset import Dataset
from src.domain.model.ClassifierModel import ClassifierModel

from src.util.numpy_util import convert_nparray_to_tensor


class DeepSHAP(BaseSelectorWeight):
    def __init__(self, n_features: int, n_labels: int, config: DatasetConfig) -> None:
        super().__init__(n_features, n_labels, config)
        self._model = ClassifierModel(n_features, n_labels, config).to(DeviceGetter.execute())
        self._k = config.shap_k
        self._representative_k = config.shap_representative_k

    def get_name() -> str:
        return "DeepSHAP"
    
    def can_predict(self) -> bool:
        return True

    def get_specificity(self) -> SelectorSpecificity:
        return SelectorSpecificity.PER_LABEL

    def fit(self, train_dataset: Dataset, test_dataset: Dataset) -> None: 
        PyTorchFit.execute(self._model, train_dataset, self._config)
        shap_samples = shap.sample(test_dataset.get_features(), self._k)
        shap_representative = shap.sample(train_dataset.get_features(), self._representative_k)
        explainer = shap.DeepExplainer(model=self._model, data=convert_nparray_to_tensor(shap_representative))
        self._shap_values = explainer.shap_values(convert_nparray_to_tensor(shap_samples))

    def predict(self, dataset: Dataset) -> np.ndarray:
        y_pred = self.predict_probabilities(dataset)
        return np.argmax(y_pred, 1)
    
    def predict_probabilities(self, dataset: Dataset, use_softmax: bool=True) -> np.ndarray:
        return PyTorchPredict.execute(self._model, dataset.get_features(), use_softmax)
    
    def get_general_weights(self) -> np.ndarray:
        return np.max(self.get_per_label_weights(), axis=0)
    
    def get_per_label_weights(self) -> np.ndarray:
        shap_values = np.abs(self._shap_values).mean(0)
        n_labels = len(shap_values[0])
        transpose = []
        for i in range(0, n_labels):
            transpose.append([])
        for feature_per_class in shap_values:
            for i in range(0, n_labels):
                transpose[i].append(feature_per_class[i])
        for i in range(0, n_labels):
            transpose[i] = np.array(transpose[i], dtype=np.float64)
        return transpose