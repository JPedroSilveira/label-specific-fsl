import statistics
from typing import Any, Dict, List, Type

from src.domain.log.Logger import Logger
from src.domain.selector.types.base.BaseSelector import BaseSelector
from src.domain.storage.ExecutionStorage import ExecutionStorage

class ReducedSetStabilityResultAggregator:
    @staticmethod
    def execute(selectors_class: List[Type[BaseSelector]], storage: ExecutionStorage) -> None:
        for selector_class in selectors_class:
            Logger.execute("Metric: Reduced Set Stability")
            Logger.execute(f"- Selector: {selector_class.get_name()}")
            executions_scores_per_size_per_label_per_metric: List[Dict[str, Dict[str, Dict[str, float]]]] = storage.get_reduced_stability_score(selector_class)
            scores_per_size_per_label_per_metric: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
            for per_size_per_label_per_metric in executions_scores_per_size_per_label_per_metric:
                for size, per_label_per_metric in per_size_per_label_per_metric.items():
                    if size not in scores_per_size_per_label_per_metric:
                        scores_per_size_per_label_per_metric[size] = { }
                    for label, per_metric in per_label_per_metric.items():
                        if label not in scores_per_size_per_label_per_metric[size]:
                            scores_per_size_per_label_per_metric[size][label] = { }
                        for metric, score in per_metric.items():     
                            if metric not in scores_per_size_per_label_per_metric[size][label]:  
                                scores_per_size_per_label_per_metric[size][label][metric] = []                         
                            scores_per_size_per_label_per_metric[size][label][metric].append(score)
            for size, scores_per_label_per_metric in scores_per_size_per_label_per_metric.items():
                Logger.execute(f"-- Reduced size: {size}")
                for label, scores_per_metric in scores_per_label_per_metric.items():
                    Logger.execute(f"--- Label: {label}")
                    for metric, scores in scores_per_metric.items():
                        mean = statistics.mean(scores)
                        stdev = statistics.stdev(scores)
                        Logger.execute(f"---- Metric: {metric}")
                        Logger.execute(f"----- Mean: {mean}")
                        Logger.execute(f"----- Standard deviation: {stdev}")
                        Logger.execute(f"----- Values: {scores}")
                            
                        
                
                    
            
            