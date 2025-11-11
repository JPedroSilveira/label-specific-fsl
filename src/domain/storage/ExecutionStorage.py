from typing import Dict, List, Type

from src.domain.selector.types.base.BaseSelector import BaseSelector


class ExecutionStorage:
    def __init__(self) -> None:
        self._execution_time_per_selector: Dict[str, List[float]] = {}
        
    def add_execution_time(self, selector: BaseSelector, value: float) -> None:
        if selector.get_selector_name() in self._execution_time_per_selector:
            self._execution_time_per_selector[selector.get_selector_name()].append(value)
        else:
            self._execution_time_per_selector[selector.get_selector_name()] = [value]
            
    def get_execution_time(self, selector_class: Type[BaseSelector]) -> List[float]:
        return self._execution_time_per_selector[selector_class.get_name()]