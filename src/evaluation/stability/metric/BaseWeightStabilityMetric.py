import numpy as np

from src.model.SelectorType import SelectorType
from src.evaluation.stability.metric.BaseStabilityMetric import BaseStabilityMetric


class BaseWeightStabilityMetric(BaseStabilityMetric):
    def should_execute(self):
        return SelectorType.WEIGHT in self._get_available_selector_types()
    
    def get_general_selection(self) -> list[np.ndarray]:
        return [self.sort_weights_by_ranking(item.get_general_weights(), item.get_general_ranking()) for item in self._history.get_items()]
    
    def get_per_class_selection(self) -> list[list[np.ndarray]]:
        selections_weight = [self.sort_weights_per_class_by_ranking(item.get_weights_per_class(), item.get_rank_per_class()) for item in self._history.get_items()]
        return self._split_selections_per_class(selections_weight)

    def sort_weights_per_class_by_ranking(self, weights_per_class, rank_per_class) -> list[np.ndarray]:
        sorted_weights_per_class = []
        for c, class_weights in enumerate(weights_per_class):
            class_rank = rank_per_class[c]
            sorted_class_weights = self.sort_weights_by_ranking(class_weights, class_rank)
            sorted_weights_per_class.append(sorted_class_weights)
        return sorted_weights_per_class

    def sort_weights_by_ranking(self, weights, rank) -> list[np.ndarray]:
        return [weights[i] for i in rank]