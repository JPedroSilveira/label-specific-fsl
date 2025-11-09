from src.config.general_config import LIME_K
import lime
import lime.lime_tabular
import numpy as np
from src.domain.data.DatasetsCreator import get_train_and_test_data_from_dataset
from src.model.Dataset import Dataset
from src.model.ClassifierModel import ClassifierModel
from src.pytorch_helpers.PyTorchFit import pytorch_fit
from src.pytorch_helpers.PyTorchPredict import pytorch_predict_propabilities
from src.selector.BaseWeightSelectorWrapper import BaseWeightSelectorWrapper
from src.selector.enum.PredictionMode import PredictionMode
from src.selector.enum.SelectionSpecificity import SelectionSpecificity
from src.util.device_util import get_device


class LIMESelectorWrapper(BaseWeightSelectorWrapper):
    def __init__(self, n_features, n_labels):
        super().__init__(n_features, n_labels)
        device = get_device()
        self._model = ClassifierModel(n_features, n_labels).to(device)
        self._n_features = n_features
        self._n_labels = n_labels
        self._k = LIME_K

    def get_name() -> str:
        return "LIME"

    def get_prediction_mode(self) -> PredictionMode:
        return PredictionMode.AVAILABLE

    def get_selection_specificities(self):
        return [SelectionSpecificity.PER_LABEL, SelectionSpecificity.GENERAL]

    def fit(self, train_dataset: Dataset, test_dataset: Dataset): 
        pytorch_fit(self._model, train_dataset)
        self._fit_selector(train_dataset, test_dataset)

    def _fit_selector(self, train_dataset: Dataset, test_dataset: Dataset):
        # Calculte how much percent of the test dataset should be used to respect k
        if self._k > len(train_dataset.get_features()):
            k_dataset = train_dataset
        else:
            k_dataset_percent = self._k / len(train_dataset.get_features())
            # Split the test dataset to get a subset that respects k
            k_dataset = get_train_and_test_data_from_dataset(train_dataset, test_size=k_dataset_percent).get_test()
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
        return pytorch_predict_propabilities(self._model, x)
    
    def predict(self, dataset: Dataset) -> np.ndarray:
        y_pred = self.predict_probabilities(dataset)
        return np.argmax(y_pred, 1)
    
    def predict_probabilities(self, dataset: Dataset, use_softmax: bool=True) -> np.ndarray:
        return pytorch_predict_propabilities(self._model, dataset.get_features(), use_softmax)
 
    def get_general_weights(self) -> np.ndarray:
        return np.max(self.get_weights_per_class(), axis=0)
    
    def get_weights_per_class(self) -> np.ndarray:
        weights_per_class = []
        for class_weights in self._feature_importance:
            weights_per_class.append(class_weights)
        return weights_per_class