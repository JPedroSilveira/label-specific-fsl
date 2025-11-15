from typing import List
import numpy as np

from config.type import DatasetConfig
from src.domain.selector.types.enum.SelectorType import SelectorType
from src.domain.selector.types.base.BaseSelector import BaseSelector
from src.domain.selector.types.enum.SelectorSpecificity import SelectorSpecificity
from src.util.rank_util import calculate_rank_from_weights


class BaseSelectorWeight(BaseSelector):
    def __init__(self, n_features: int, n_labels: int, config: DatasetConfig) -> None:
        super().__init__(n_features, n_labels, config)
        self._general_ranking = None
        self._per_class_ranking = None

    def get_type(self) -> SelectorType:
        return SelectorType.WEIGHT
    
    def get_general_ranking(self) -> np.ndarray:
        if self._general_ranking is None:
            self._general_ranking = calculate_rank_from_weights(self.get_general_weights())
   
        return self._general_ranking

    def get_per_label_ranking(self) -> List[np.ndarray]:
        if self._per_class_ranking is None:
            self._per_class_ranking = [calculate_rank_from_weights(weights) for weights in self.get_weights_per_label()]
   
        return self._per_class_ranking