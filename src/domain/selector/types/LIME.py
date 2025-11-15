import lime
import lime.lime_tabular
import numpy as np

from config.type import DatasetConfig
from src.domain.pytorch.PyTorchPredict import PyTorchPredict
from src.domain.pytorch.PyTorchFit import PyTorchFit
from src.domain.device.DeviceGetter import DeviceGetter
from src.domain.data.types.Dataset import Dataset
from src.domain.selector.types.base.BaseSelector import SelectorSpecificity
from src.domain.selector.types.base.BaseSelectorWeight import BaseSelectorWeight
from src.domain.model.ClassifierModel import ClassifierModel
from src.domain.data.DatasetSplitter import DatasetSplitter


class LIME(BaseSelectorWeight):
    def __init__(self, n_features, n_labels, config: DatasetConfig) -> None:
        super().__init__(n_features, n_labels, config)
        self._model = ClassifierModel(n_features, n_labels, config).to(DeviceGetter.execute())
        self._n_features = n_features
        self._n_labels = n_labels
        self._k = config.lime_k
        self._config = config

    def get_name() -> str:
        return "LIME"
    
    def can_predict(self) -> bool:
        return True
    
    def get_specificity(self) -> SelectorSpecificity:
        return SelectorSpecificity.PER_LABEL

    def fit(self, train_dataset: Dataset, test_dataset: Dataset) -> None: 
        PyTorchFit.execute(self._model, train_dataset, self._config)
        self._fit_selector(train_dataset, test_dataset)

    def _fit_selector(self, train_dataset: Dataset, test_dataset: Dataset) -> None:
        # Calculte how much percent of the test dataset should be used to respect k
        if self._k > len(train_dataset.get_features()):
            k_dataset = train_dataset
        else:
            k_dataset_percent = self._k / len(train_dataset.get_features())
            # Split the test dataset to get a subset that respects k
            k_dataset = DatasetSplitter.execute(train_dataset, k_dataset_percent, self._config).get_test()
        explainer = lime.lime_tabular.LimeTabularExplainer(
            mode='classification',
            training_data=train_dataset.get_features(), 
            training_labels=train_dataset.get_labels(),
            feature_names=list(range(0, self._n_features)),
            class_names=list(range(0, self._n_labels))
        )
        feature_importance_by_label = []
        count_by_label = []
        for _ in range(0, self._n_labels):
            feature_importance = []
            for _ in range(0, self._n_features):
                feature_importance.append(0)
            feature_importance_by_label.append(feature_importance)
            count_by_label.append(0)
        for i, row in enumerate(k_dataset.get_features()):
            label = k_dataset.get_labels()[i]
            explanation = explainer.explain_instance(
                data_row=row, 
                predict_fn=self.lime_predict,
                num_features=self._n_features,
                num_samples=100,
                labels=list(range(0, self._n_labels))
            )
            for feature_explanation in explanation.as_map()[label]:
                feature = feature_explanation[0]
                importance = np.abs(feature_explanation[1])
                feature_importance_by_label[label][feature] += importance
            count_by_label[label] += 1
        for label in range(0, self._n_labels):
            label_importance = np.array(feature_importance_by_label[label])
            feature_importance_by_label[label] = label_importance / count_by_label[label]
        self._feature_importance = np.array(feature_importance_by_label)

    def lime_predict(self, x) -> np.ndarray:
        return PyTorchPredict.execute(self._model, x)
    
    def predict(self, dataset: Dataset) -> np.ndarray:
        y_pred = self.predict_probabilities(dataset)
        return np.argmax(y_pred, 1)
    
    def predict_probabilities(self, dataset: Dataset, use_softmax: bool=True) -> np.ndarray:
        return PyTorchPredict.execute(self._model, dataset.get_features(), use_softmax)
 
    def get_general_weights(self) -> np.ndarray:
        return np.max(self.get_per_label_weights(), axis=0)
    
    def get_per_label_weights(self) -> np.ndarray:
        weights_per_class = []
        for class_weights in self._feature_importance:
            weights_per_class.append(class_weights)
        return weights_per_class