import numpy as np
from sklearn.neighbors import NearestNeighbors

from config.type import DatasetConfig
from src.domain.selector.types.base.BaseSelector import SelectorSpecificity
from src.domain.selector.types.base.BaseSelectorWeight import BaseSelectorWeight
from src.model.Dataset import Dataset


class ReliefF(BaseSelectorWeight):
    def __init__(self, n_features: int, n_labels: int, config: DatasetConfig) -> None:
        super().__init__(n_features, n_labels)
        self._weights = None
        self._n_neighbors = config.relieff_k

    def get_name() -> str:
        return "ReliefF"
    
    def can_predict(self) -> bool:
        return False

    def get_specificity(self) -> SelectorSpecificity:
        return SelectorSpecificity.GENERAL

    def fit(self, train_dataset: Dataset, test_dataset: Dataset) -> None: 
        X = train_dataset.get_features()
        y = train_dataset.get_labels()
        n_samples, n_features = X.shape
        self._weights = np.zeros(n_features)
        _, neighbors = NearestNeighbors(metric='manhattan').fit(X).kneighbors(X)
        classes, classes_counts = np.unique(y, return_counts=True)
        class_prob = {classes[n]: x for n, x in enumerate(classes_counts / n_samples)}
        class_factor = {k: {c: class_prob[k] / (1 - class_prob[c]) for c in classes if c != k} for k in classes}
        for count in classes_counts:
            if count < self._n_neighbors:
                self._n_neighbors = np.min(classes_counts)
        instances_by_class = {x: np.arange(n_samples)[y == x] for x in classes}
        for instance in range(n_samples):
            instance_class = y[instance]
            instance_neighbors = neighbors[instance][1:]
            hits = self._n_first_x_in_y(instance_neighbors, instances_by_class[instance_class])
            misses_by_class = {
                c: self._n_first_x_in_y(instance_neighbors, instances_by_class[c])
                for c in classes
                if c != instance_class
            }
            for hit in hits:
                self._weights -= np.array(abs(X[instance] - X[hit])) / self._n_neighbors
            for (c, misses) in misses_by_class.items():
                for miss in misses:
                    class_weight = class_factor[instance_class][c]
                    miss_weights = np.array(abs(X[instance] - X[miss])) / (self._n_neighbors * class_weight)
                    self._weights += miss_weights
    
    def get_general_weights(self) -> np.ndarray:
        return self._weights
    
    def _n_first_x_in_y(self, x, y) -> list:
        num_x = len(x)
        i, first_k = 0, []
        while len(first_k) < self._n_neighbors and i < num_x:
            if x[i] in y:
                first_k.append(x[i])
            i += 1
        return first_k