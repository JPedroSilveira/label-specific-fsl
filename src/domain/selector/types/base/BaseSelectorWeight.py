import numpy as np

from src.model.SelectorType import SelectorType
from src.domain.selector.types.base.BaseSelector import BaseSelector
from src.model.SelectorSpecificity import SelectorSpecificity
from src.util.rank_util import calculate_rank_from_weights


class BaseSelectorWeight(BaseSelector):
    def __init__(self, n_features: int, n_labels: int) -> None:
        super().__init__(n_features, n_labels)
        self._general_ranking = None
        self._per_class_ranking = None

    def get_type(self) -> SelectorType:
        return SelectorType.WEIGHT
    
    def get_general_ranking(self) -> np.ndarray:
        if self._general_ranking is None and SelectorSpecificity.GENERAL in self.get_specificity():
            self._general_ranking = calculate_rank_from_weights(self.get_general_weights())
   
        return self._general_ranking

    def get_ranking_per_class(self) -> list[np.ndarray]:
        if self._per_class_ranking is None and SelectorSpecificity.PER_LABEL in self.get_specificity():
            self._per_class_ranking = [calculate_rank_from_weights(weights) for weights in self.get_weights_per_class()]
   
        return self._per_class_ranking