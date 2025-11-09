import numpy as np
from src.evaluation.stability.metric.BaseStabilityMetric import BaseStabilityMetric
from src.selector.enum.SelectionMode import SelectionMode


class BaseRankStabilityMetric(BaseStabilityMetric):
    def should_execute(self):
        return SelectionMode.RANK in self._get_available_selecion_modes()
    
    def get_general_selection(self) -> list[np.ndarray]:
        return [item.get_general_ranking() for item in self._src.history.get_items()]

    def get_per_class_selection(self) -> list[list[np.ndarray]]:
        selections_rank = [item.get_rank_per_class() for item in self._src.history.get_items()]
        return self._split_selections_per_class(selections_rank)