import numpy as np
from scipy import stats
from sklearn.feature_selection import SelectKBest
from src.model.Dataset import Dataset
from src.selector.BaseWeightSelectorWrapper import BaseWeightSelectorWrapper
from src.selector.enum.PredictionMode import PredictionMode
from src.selector.enum.SelectionSpecificity import SelectionSpecificity


class KruskalWallisSelectorWrapper(BaseWeightSelectorWrapper):
    def __init__(self, n_features, n_labels):
        super().__init__(n_features, n_labels)
        self.model = SelectKBest(score_func=self._kruskal_wallis, k='all')

    def get_name() -> str:
        return "KruskalWallis"

    def get_prediction_mode(self) -> PredictionMode:
        return PredictionMode.UNAVAILABLE
    
    def get_selection_specificities(self):
        return [SelectionSpecificity.GENERAL]

    def fit(self, train_dataset: Dataset, _: Dataset): 
        X = train_dataset.get_features()
        y = train_dataset.get_labels()
        self.src.model.fit(X, y)
    
    def get_general_weights(self) -> np.ndarray:
        return self.src.model.scores_
    
    def _single_feature_kruskal_wallis(self, X, y):
        return stats.kruskal(*[X[y == c] for c in np.unique(y)])
    
    def _kruskal_wallis(self, X, y):
        if len(X.shape) == 1:
            return self._single_feature_kruskal_wallis(X, y)
        results = np.array([self._single_feature_kruskal_wallis(x, y) for x in X.T])
        scores, pvalues = results.T
        return scores, pvalues