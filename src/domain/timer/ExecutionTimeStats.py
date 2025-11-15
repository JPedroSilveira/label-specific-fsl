import statistics
from typing import List, Type

from src.domain.log.Logger import Logger
from src.domain.selector.types.base.BaseSelector import BaseSelector
from src.domain.storage.ExecutionStorage import ExecutionStorage


class ExecutionTimeStats:
    @staticmethod
    def execute(selectors_class: List[Type[BaseSelector]], storage: ExecutionStorage) -> None:
        Logger.execute("Metric: Execution time")
        for selector_class in selectors_class:
            Logger.execute(f"- Selector: {selector_class.get_name()}")
            execution_times = storage.get_execution_time(selector_class)
            mean = statistics.mean(execution_times)
            stdev = statistics.stdev(execution_times)
            Logger.execute(f"-- Mean: {mean}")
            Logger.execute(f"-- Standard deviation: {stdev}")
            Logger.execute(f"-- Values: {execution_times}")
            