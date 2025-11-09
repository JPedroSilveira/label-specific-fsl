

class StabilityScore:
    def __init__(self, metric_name: str, selection_size: int, score: float, selector_name: str, target_label: str = "General"):
        self._metric_name = metric_name
        self._selection_size = selection_size
        self._score = score
        self._target_label = target_label
        self._selector_name = selector_name
    
    def get_metric(self):
        return self._metric_name

    def get_selection_size(self):
        return self._selection_size
    
    def get_score(self):
        return self._score
    
    def get_target_label(self):
        return self._target_label
    
    def get_selector_name(self):
        return self._selector_name
    
    def __str__(self) -> str:
        return f"Metric: {self._metric_name}  \n    Target Label: {self._target_label} \n    Selection Size: {self._selection_size} \n    Score: {self._score} \n    Selector: {self._selector_name}"