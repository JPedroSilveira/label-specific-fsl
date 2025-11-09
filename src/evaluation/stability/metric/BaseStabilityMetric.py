from typing import List
from itertools import combinations
import numpy as np

from src.model.SelectorType import SelectorType
from src.model.SelectorSpecificity import SelectorSpecificity
from src.history.ExecutionHistory import ExecutionHistory
from src.evaluation.stability.StabilityScore import StabilityScore


class BaseStabilityMetric():
    def __init__(self, history: ExecutionHistory, selection_size: int):
        self._history = history
        self._selection_size = selection_size

    def get_name() -> str:
        """
        Returns the metric name
        """
        raise NotImplemented("")
    
    def get_class_name(self):
        return self.__class__.get_name()

    def should_execute(self) -> bool:
        """
        Define if a metric should be calculated based on ExecutionHistory
        """
        raise NotImplemented("")
    
    def get_general_selection(self) -> list[np.ndarray]:
        """
        Return the results for each execution based on the data used by the metric (rank, weight, ...)
        """
        raise NotImplemented("")

    def get_per_class_selection(self) -> list[list[np.ndarray]]:
        """
        Return the results for each class for each execution based on the data used by the metric (rank, weight, ...)
        """
        raise NotImplemented("")
    
    def calculate_metric(self, result1: np.ndarray, result2: np.ndarray) -> float:
        """
        Returns the metric score based on two results from two executions
        """
        raise NotImplemented("")

    def calculate_general(self, selector_name: str) -> StabilityScore:
        selections = self.get_general_selection()
        top_n_from_selections = self._get_top_n_from_selections(selections) 
        score = self._calculate_mean_score_two_by_two(top_n_from_selections)
        return StabilityScore(self.get_class_name(), self._selection_size, score, selector_name)
    
    def calculate_per_class(self, selector_name: str) -> list[StabilityScore]:
        scores_per_class = []
        labels = self._get_labels()
        per_class_selections = self.get_per_class_selection()
        for index, selections in enumerate(per_class_selections):
            top_n_from_selections = self._get_top_n_from_selections(selections)  
            score = self._calculate_mean_score_two_by_two(top_n_from_selections)
            scores_per_class.append(StabilityScore(self.get_class_name(), self._selection_size, score, selector_name, target_label=labels[index]))
        return scores_per_class
    
    def _get_top_n_from_selections(self, selections: List[np.ndarray]):
        top_n_from_selections = []
        for selection in selections:
            top_n_from_selections.append(selection[:self._selection_size])  
        return top_n_from_selections
    
    def _calculate_mean_score_two_by_two(self, selections) -> float:
        return np.mean([
            self.calculate_metric(selection1, selection2)
            for selection1, selection2 in combinations(selections, 2)
        ])
    
    def _get_labels(self):
        return self._history.get_labels()
    
    def _get_available_selector_types(self) -> List[SelectorType]:
        return self._history.get_available_seletion_modes()
    
    def _get_available_selectors_specificity(self) -> List[SelectorSpecificity]:
        return self._history.get_selection_specificities()
    
    def _split_selections_per_class(self, selections):
        selections_per_class = [[] for _ in self._get_labels()]
        for selection in selections:
            for c in self._get_labels():
                selections_per_class[c].append(selection[c])
        return selections_per_class