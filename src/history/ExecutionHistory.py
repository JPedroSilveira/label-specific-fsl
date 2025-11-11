import numpy as np
from typing import List

from src.domain.selector.types.enum.SelectorSpecificity import SelectorSpecificity
from src.domain.selector.types.enum.SelectorType import SelectorType
from src.domain.selector.types.base.BaseSelector import BaseSelector
from src.domain.data.types.SplittedDataset import SplittedDataset
from src.evaluation.prediction.PredictionScore import SelectorPredictionScore

class HistoryItem():
    def __init__(self, selector: BaseSelector, execution_time: float, prediction_score: SelectorPredictionScore | None) -> None:
        self._selector = selector
        self._execution_time = execution_time
        self._prediction_score = prediction_score
        self._weights = None
        self._weights_per_class = None
        self._rank = None
        self._rank_per_class = None
        self._extract_selection(selector)

    def get_selector(self) -> BaseSelector:
        return self._selector
    
    def get_prediction_score(self) -> SelectorPredictionScore | None:
        return self._prediction_score
    
    def get_execution_time(self) -> float:
        return self._execution_time

    def get_available_seletion_modes(self) -> list[SelectorSpecificity]:
        available_selection_modes = []
        if self.get_general_weights() is not None or self.get_weights_per_class() is not None:
            available_selection_modes.append(SelectorSpecificity.WEIGHT)
        if self.get_general_ranking() is not None or self.get_rank_per_class() is not None:
            available_selection_modes.append(SelectorSpecificity.RANK)
        return available_selection_modes
    
    def get_available_selection_specifities(self) -> list[SelectorSpecificity]:
        available_selection_specificities = []
        if self.get_general_ranking() is not None or self.get_general_weights() is not None:
            available_selection_specificities.append(SelectorSpecificity.GENERAL)
        if self.get_rank_per_class() is not None or self.get_weights_per_class() is not None:
            available_selection_specificities.append(SelectorSpecificity.PER_LABEL)
        return available_selection_specificities

    def get_general_weights(self) -> None | np.ndarray:
        return self._weights
    
    def get_weights_per_class(self) -> None | list[np.ndarray]:
        return self._weights_per_class

    def get_general_ranking(self) -> None | np.ndarray:
        return self._rank
    
    def get_rank_per_class(self) -> None | list[np.ndarray]:
        return self._rank_per_class        

    def _extract_selection(self, selector: BaseSelector):
        selector_mode = selector.get_type()
        selector_specificities = selector.get_selection_specificities()
        if selector_mode == SelectorType.WEIGHT:
            if SelectorSpecificity.GENERAL in selector_specificities:
                self._weights = selector.get_general_weights()
                self._rank = selector.get_general_ranking()
            if SelectorSpecificity.PER_LABEL in selector_specificities:
                self._weights_per_class = selector.get_weights_per_label()
                self._rank_per_class = selector.get_ranking_per_class()
        elif selector_mode == SelectorType.RANK:
            if SelectorSpecificity.GENERAL in selector_specificities:
                self._rank = selector.get_general_ranking()
            if SelectorSpecificity.PER_LABEL in selector_specificities:
                self._rank_per_class = selector.get_ranking_per_class()


class ExecutionHistory():
    def __init__(self) -> None:
        self._items: List[HistoryItem] = []
        self._selector = None
        self._selector_name = None
        self._selection_mode = None
        self._selection_specificities = None
        self._labels = None
        self._n_labels = None
        self._n_features = None
        self._start_time = None

    def get_items(self) -> List[HistoryItem]:
        return self._items
    
    def get_selector_name(self) -> str:
        self._verify_if_history_was_initialized()
        return self._selector_name
    
    def get_available_selector_types(self) -> List[SelectorType]: 
        self._verify_if_history_was_initialized()
        return self._items[0].get_available_seletion_modes()
    
    def get_selectors_specificity(self) -> List[SelectorSpecificity]:
        self._verify_if_history_was_initialized()
        return self._items[0].get_available_selection_specifities()
    
    def get_labels(self) -> None | List:
        self._verify_if_history_was_initialized()
        return self._labels
    
    def get_n_labels(self) -> None | List:
        self._verify_if_history_was_initialized()
        return self._n_labels
    
    def get_n_features(self) -> None | List:
        self._verify_if_history_was_initialized()
        return self._n_features
    
    def _verify_if_history_was_initialized(self) -> None:
        if self._selector is None:
            raise RuntimeError("Trying to fetch information from a history without any added item")
    
    def add(self, selector: BaseSelector, dataset: SplittedDataset, execution_time: float, prediction_score: SelectorPredictionScore | None) -> None:
        if self._selector is None:
            self._selector = selector
            self._selector_name = selector.get_selector_name()
            self._labels = dataset.get_label_types()
            self._n_labels = dataset.get_n_labels()
            self._n_features = dataset.get_n_features()
            self._selection_mode = selector.get_selection_mode()
            self._selection_specificities = selector.get_selection_specificities()
        elif type(self._selector) != type(selector):
            raise ValueError("Execution history can't have items from different types of selector")
        new_item = HistoryItem(selector, execution_time, prediction_score)
        self._items.append(new_item)