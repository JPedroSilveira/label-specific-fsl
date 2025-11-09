import time
from src.util.print_util import print_with_time


class ExecutionTimeCounter():
    def __init__(self) -> None:
        self._start_time = 0.0
        self._end_time = 0.0

    def start(self):
        if self._start_time == 0.0:
            self._start_time = time.perf_counter()
        else:
            raise ValueError('Execution timer was already started! Create a new instance for a new counter.')
        return self
    
    def print_start(self, name):
        print_with_time(name)
        self.start()
        return self
    
    def end(self):
        self._end_time = time.perf_counter()
        return self

    def get_execution_time(self):
        return self._end_time - self._start_time

    def print_end(self, name):
        if self._end_time == 0.0:
            self.end()
        print_with_time(f'{name} took {self.get_execution_time()}s')
        return self
