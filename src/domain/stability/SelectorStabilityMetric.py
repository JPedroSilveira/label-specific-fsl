import os
import pandas as pd
import numpy as np
from typing import Dict, List, Type
from itertools import combinations

from config.type import Config
from src.domain.data.reader.WeightReader import WeightReader
from src.domain.stability.StabilityMetricTypeCreator import StabilityMetricTypeCreator
from src.domain.log.Logger import Logger
from src.domain.selector.types.base.BaseSelector import BaseSelector


class SelectorStabilityMetric:
    @staticmethod
    def execute(selectors_class: List[Type[BaseSelector]], config: Config) -> None:
        metrics = StabilityMetricTypeCreator.execute(config)
        weights_per_selector_and_specificity: Dict[str, Dict[str,List[pd.DataFrame]]] = {}
        for selector_class in selectors_class:
            weights_per_selector_and_specificity[selector_class.get_name()] = WeightReader.execute(selector_class, config)
        for metric in metrics:
            Logger.execute(f"Metric: {metric.get_name()}")
            for selector_class in selectors_class:
                weights_per_specificity = weights_per_selector_and_specificity[selector_class.get_name()]
                Logger.execute(f"- Selector: {selector_class.get_name()}")
                for specificity in weights_per_specificity.keys():
                    weights = weights_per_specificity[specificity]
                    value = np.mean([
                        metric.execute(weights1, weights2, config)
                        for weights1, weights2 in combinations(weights, 2)
                    ])
                    Logger.execute(f"-- Label {specificity}: {value}")            