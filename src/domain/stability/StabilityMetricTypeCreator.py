from typing import List, Type

from config.type import Config
from src.domain.stability.types.base.BaseStabilityMetric import BaseStabilityMetric


class StabilityMetricTypeCreator:
    @staticmethod
    def execute(config: Config) -> List[Type[BaseStabilityMetric]]:
        metrics = set([stability.name for stability in config.stability_metric])
        metrics_class = []
        for metric in metrics:
            module = __import__('src.domain.stability.types.' + metric, fromlist=[metric])
            metric_class = getattr(module, metric)
            metrics_class.append(metric_class)
        return metrics_class