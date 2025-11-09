import numpy as np
from src.selector.BaseSelectorWrapper import BaseSelectorWrapper
from src.selector.enum.SelectionSpecificity import SelectionSpecificity
from src.selector.enum.SelectionMode import SelectionMode
from src.util.rank_util import calculate_rank_from_weights


class BaseWeightSelectorWrapper(BaseSelectorWrapper):
    def __init__(self, n_features, n_labels) -> None:
        super().__init__(n_features, n_labels)
        self._general_ranking = None
        self._per_class_ranking = None

    def get_selection_mode(self):
        return SelectionMode.WEIGHT
    
    def get_general_ranking(self) -> np.ndarray:
        if self._general_ranking is None and SelectionSpecificity.GENERAL in self.get_selection_specificities():
            self._general_ranking = calculate_rank_from_weights(self.get_general_weights())
   
        return self._general_ranking

    def get_ranking_per_class(self) -> list[np.ndarray]:
        if self._per_class_ranking is None and SelectionSpecificity.PER_LABEL in self.get_selection_specificities():
            self._per_class_ranking = [calculate_rank_from_weights(weights) for weights in self.get_weights_per_class()]
   
        return self._per_class_ranking