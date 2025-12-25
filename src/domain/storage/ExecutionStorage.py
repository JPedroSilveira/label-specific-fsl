from typing import Dict, List, Type

from src.domain.selector.types.base.BaseSelector import BaseSelector
from src.domain.prediction.types.ClassificationScoreGeneralReport import ClassificationScoreGeneralReport
from src.domain.prediction.types.ClassificationScoreLabelReport import ClassificationScoreLabelReport

class ExecutionStorage:
    def __init__(self) -> None:
        self._execution_time_per_selector: Dict[str, List[float]] = {}
        self._reduced_stability_score_per_selector_per_size_per_label_per_metric: Dict[str, List[Dict[str, Dict[str, Dict[str, float]]]]] = {}
        self._general_prediction_score_per_selector: Dict[str, List[ClassificationScoreGeneralReport]] = {}
        self._per_label_prediction_score_per_selector: Dict[str, Dict[int, List[ClassificationScoreLabelReport]]] = {}
        
    def add_execution_time(self, selector: BaseSelector, value: float) -> None:
        if selector.get_selector_name() in self._execution_time_per_selector:
            self._execution_time_per_selector[selector.get_selector_name()].append(value)
        else:
            self._execution_time_per_selector[selector.get_selector_name()] = [value]
            
    def get_execution_time(self, selector_class: Type[BaseSelector]) -> List[float]:
        return self._execution_time_per_selector[selector_class.get_name()]
    
    def add_reduced_stability_score(self, selector: BaseSelector, value: Dict[str, Dict[str, Dict[str, float]]]) -> None:
        if selector.get_selector_name() in self._reduced_stability_score_per_selector_per_size_per_label_per_metric:
            self._reduced_stability_score_per_selector_per_size_per_label_per_metric[selector.get_selector_name()].append(value)
        else:
            self._reduced_stability_score_per_selector_per_size_per_label_per_metric[selector.get_selector_name()] = [value]
            
    def get_reduced_stability_score(self, selector_class: Type[BaseSelector]) -> List[Dict[str, Dict[str, Dict[str, float]]]]:
        return self._reduced_stability_score_per_selector_per_size_per_label_per_metric[selector_class.get_name()]
    
    def add_general_prediction_score(self, selector: BaseSelector, value: ClassificationScoreGeneralReport) -> None:
        if selector.get_selector_name() in self._general_prediction_score_per_selector:
            self._general_prediction_score_per_selector[selector.get_selector_name()].append(value)
        else:
            self._general_prediction_score_per_selector[selector.get_selector_name()] = [value]
            
    def get_general_prediction_score(self, selector_class: Type[BaseSelector]) -> List[ClassificationScoreGeneralReport]:
        if selector_class.get_name() not in self._general_prediction_score_per_selector:
            return None
        return self._general_prediction_score_per_selector[selector_class.get_name()]
            
    def add_per_label_prediction_score(self, selector: BaseSelector, label: int, value: ClassificationScoreLabelReport) -> None:
        if selector.get_selector_name() not in self._per_label_prediction_score_per_selector:
            self._per_label_prediction_score_per_selector[selector.get_selector_name()] = {}
        if label in self._per_label_prediction_score_per_selector[selector.get_selector_name()]:
            self._per_label_prediction_score_per_selector[selector.get_selector_name()][label].append(value)
        else:
            self._per_label_prediction_score_per_selector[selector.get_selector_name()][label] = [value]
            
    def get_per_label_prediction_scores(self, selector_class: Type[BaseSelector]) -> Dict[int, List[ClassificationScoreLabelReport]]:
        if selector_class.get_name() not in self._per_label_prediction_score_per_selector:
            return None
        return self._per_label_prediction_score_per_selector[selector_class.get_name()]
            
    