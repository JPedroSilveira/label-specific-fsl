import numpy as np
from sklearn import preprocessing


class Dataset():
    def __init__(self, features: np.ndarray, labels: np.ndarray, label_types: list, feature_names: list[str], informative_features: list[int], informative_features_per_label: dict[int, list[int]]):
        if len(features) == 0 or len(labels) == 0:
            raise ValueError("Number of features and labels should be greater than zero")
        self._features = features
        self._labels = labels
        self._label_types = label_types
        self._feature_names = feature_names
        self._informative_features = informative_features
        self._informative_features_per_label = informative_features_per_label

    def get_features(self) -> np.ndarray:
        return self._features
    
    def get_encoded_labels(self):
        labels = np.array(self.get_labels()).reshape(-1, 1)
        one_hot_encoder = preprocessing.OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(labels)
        return one_hot_encoder.transform(labels)
    
    def get_labels(self) -> np.ndarray:
        return self._labels
    
    def get_n_features(self):
        return len(self._features[0])
    
    def get_n_labels(self):
        return len(self._label_types)
    
    def get_label_types(self):
        return self._label_types
    
    def get_feature_names(self):
        return self._feature_names
    
    def get_informative_features(self):
        return self._informative_features
    
    def get_informative_features_per_label(self):
        return self._informative_features_per_label