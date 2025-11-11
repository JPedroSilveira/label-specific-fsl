import time
from typing import Self

from src.domain.log.Logger import Logger


class ExecutionTimeCounter:
    def __init__(self) -> None:
        self._start_time = 0.0
        self._end_time = 0.0

    def start(self) -> Self:
        if self._start_time == 0.0:
            self._start_time = time.perf_counter()
        else:
            raise ValueError('Execution timer was already started! Create a new instance for a new counter.')
        return self
    
    def print_start(self, name) -> Self:
        Logger.execute(name)
        self.start()
        return self
    
    def end(self) -> Self:
        self._end_time = time.perf_counter()
        return self

    def get_execution_time(self) -> float:
        if self._end_time == 0.0:
            self.end()
        return self._end_time - self._start_time

    def print_end(self, name) -> Self:
        if self._end_time == 0.0:
            self.end()
        Logger.execute(f'{name} took {self.get_execution_time()}s')
        return self
