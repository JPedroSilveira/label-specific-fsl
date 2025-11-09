import numpy as np
from src.data.DataSplitter import SplittedDataset
from src.evaluation.prediction.PredictionScore import SelectorPredictionScore
from src.selector.BaseSelectorWrapper import BaseSelectorWrapper
from src.selector.enum.SelectionMode import SelectionMode
from src.selector.enum.SelectionSpecificity import SelectionSpecificity

class HistoryItem():
    def __init__(self, selector: BaseSelectorWrapper, execution_time: float, prediction_score: SelectorPredictionScore | None):
        self._selector = selector
        self._execution_time = execution_time
        self._prediction_score = prediction_score
        self._weights = None
        self._weights_per_class = None
        self._rank = None
        self._rank_per_class = None
        self._extract_selection(selector)

    def get_selector(self):
        return self._selector
    
    def get_prediction_score(self):
        return self._prediction_score
    
    def get_execution_time(self):
        return self._execution_time

    def get_available_seletion_modes(self) -> list[SelectionMode]:
        available_selection_modes = []
        if self.get_general_weights() is not None or self.get_weights_per_class() is not None:
            available_selection_modes.append(SelectionMode.WEIGHT)
        if self.get_general_ranking() is not None or self.get_rank_per_class() is not None:
            available_selection_modes.append(SelectionMode.RANK)

        return available_selection_modes
    
    def get_available_selection_specifities(self) -> list[SelectionSpecificity]:
        available_selection_specificities = []
        if self.get_general_ranking() is not None or self.get_general_weights() is not None:
            available_selection_specificities.append(SelectionSpecificity.GENERAL)
        if self.get_rank_per_class() is not None or self.get_weights_per_class() is not None:
            available_selection_specificities.append(SelectionSpecificity.PER_LABEL)
        return available_selection_specificities

    def get_general_weights(self) -> None | np.ndarray:
        return self._weights
    
    def get_weights_per_class(self) -> None | list[np.ndarray]:
        return self._weights_per_class

    def get_general_ranking(self) -> None | np.ndarray:
        return self._rank
    
    def get_rank_per_class(self) -> None | list[np.ndarray]:
        return self._rank_per_class        

    def _extract_selection(self, selector: BaseSelectorWrapper):
        selector_mode = src.selector.get_selection_mode()
        selector_specificities = src.selector.get_selection_specificities()
        if selector_mode == SelectionMode.WEIGHT:
            if SelectionSpecificity.GENERAL in selector_specificities:
                self._weights = src.selector.get_general_weights()
                self._rank = src.selector.get_general_ranking()
            if SelectionSpecificity.PER_LABEL in selector_specificities:
                self._weights_per_class = src.selector.get_weights_per_class()
                self._rank_per_class = src.selector.get_ranking_per_class()
        elif selector_mode == SelectionMode.RANK:
            if SelectionSpecificity.GENERAL in selector_specificities:
                self._rank = src.selector.get_general_ranking()
            if SelectionSpecificity.PER_LABEL in selector_specificities:
                self._rank_per_class = src.selector.get_ranking_per_class()


class ExecutionHistory():
    def __init__(self):
        self._items: list[HistoryItem] = []
        self._selector = None
        self._selector_name = None
        self._selection_mode = None
        self._selection_specificities = None
        self._labels = None
        self._n_labels = None
        self._n_features = None
        self._start_time = None

    def get_items(self):
        return self._items
    
    def get_selector_name(self) -> str:
        self._verify_if_history_was_initialized()
        return self._selector_name
    
    def get_available_seletion_modes(self) -> list[SelectionMode]: 
        self._verify_if_history_was_initialized()
        return self._items[0].get_available_seletion_modes()
    
    def get_selection_specificities(self) -> list[SelectionSpecificity]:
        self._verify_if_history_was_initialized()
        return self._items[0].get_available_selection_specifities()
    
    def get_labels(self):
        self._verify_if_history_was_initialized()
        return self._labels
    
    def get_n_labels(self):
        self._verify_if_history_was_initialized()
        return self._n_labels
    
    def get_n_features(self):
        self._verify_if_history_was_initialized()
        return self._n_features
    
    def _verify_if_history_was_initialized(self):
        if self._selector is None:
            raise RuntimeError("Trying to fetch information from a history without any added item")
    
    def add(self, selector: BaseSelectorWrapper, dataset: SplittedDataset, execution_time: float, prediction_score: SelectorPredictionScore | None):
        if self._selector is None:
            self._selector = selector
            self._selector_name = src.selector.get_class_name()
            self._labels = dataset.get_label_types()
            self._n_labels = dataset.get_n_labels()
            self._n_features = dataset.get_n_features()
            self._selection_mode = src.selector.get_selection_mode()
            self._selection_specificities = src.selector.get_selection_specificities()
        elif type(self._selector) != type(selector):
            raise ValueError("Execution history can't have items from different types of selector")
        new_item = HistoryItem(selector, execution_time, prediction_score)
        self._items.append(new_item)