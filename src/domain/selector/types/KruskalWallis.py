from typing import Any
import numpy as np
from scipy import stats
from sklearn.feature_selection import SelectKBest

from config.type import DatasetConfig
from src.model.SelectorSpecificity import SelectorSpecificity
from src.domain.selector.types.base.BaseSelectorWeight import BaseSelectorWeight
from src.model.Dataset import Dataset



class KruskalWallis(BaseSelectorWeight):
    def __init__(self, n_features: int, n_labels: int, config: DatasetConfig) -> None:
        super().__init__(n_features, n_labels)
        self.model = SelectKBest(score_func=self._kruskal_wallis, k='all')

    def get_name() -> str:
        return "Kruskal Wallis"
    
    def can_predict(self) -> bool:
        return False

    def get_specificity(self) -> SelectorSpecificity:
        return SelectorSpecificity.GENERAL

    def fit(self, train_dataset: Dataset, _: Dataset) -> None: 
        X = train_dataset.get_features()
        y = train_dataset.get_labels()
        self.src.model.fit(X, y)
    
    def get_general_weights(self) -> np.ndarray:
        return self.src.model.scores_
    
    def _single_feature_kruskal_wallis(self, X, y) -> Any:
        return stats.kruskal(*[X[y == c] for c in np.unique(y)])
    
    def _kruskal_wallis(self, X, y) -> Any | tuple[Any, Any]:
        if len(X.shape) == 1:
            return self._single_feature_kruskal_wallis(X, y)
        results = np.array([self._single_feature_kruskal_wallis(x, y) for x in X.T])
        scores, pvalues = results.T
        return scores, pvalues