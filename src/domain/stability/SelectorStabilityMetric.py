import os
import pandas as pd
import numpy as np
from typing import Dict, List, Type
from itertools import combinations

from config.type import Config
from src.domain.log.Logger import Logger
from src.domain.selector.types.base.BaseSelector import BaseSelector
from src.domain.stability.types.base.BaseStabilityMetric import BaseStabilityMetric


class SelectorStabilityMetric:
    @staticmethod
    def execute(selectors_class: List[Type[BaseSelector]], config: Config) -> None:
        metrics = SelectorStabilityMetric._get_metrics(config)
        weights_per_selector_and_specificity: Dict[str, Dict[str,List[pd.DataFrame]]] = {}
        for selector_class in selectors_class:
            weights_per_selector_and_specificity[selector_class.get_name()] = SelectorStabilityMetric._read_selector_weights(selector_class, config)
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
            
    @staticmethod
    def _read_selector_weights(selectors_class: Type[BaseSelector], config: Config) -> Dict[str,List[pd.DataFrame]]:
        weightsPerSpecificit: Dict[str,List[pd.DataFrame]] = {}
        all_selection_files: List[str] = os.listdir(config.output.execution_output.raw_selection)
        for selection_file in all_selection_files:
            if selection_file.endswith("_raw.csv"):
                filename_parts = selection_file.split('_')
                selector_type = filename_parts[1]
                specificit = filename_parts[2]
                if selector_type == selectors_class.get_name().lower():
                    new_weights = SelectorStabilityMetric._read_weights_from_file(selection_file, config)
                    if specificit in weightsPerSpecificit:
                        weightsPerSpecificit[specificit].append(new_weights)
                    else:
                        weightsPerSpecificit[specificit] = [new_weights]
        return weightsPerSpecificit
        
    @staticmethod
    def _read_weights_from_file(filename: str, config: Config) -> pd.DataFrame:
        return pd.read_csv(f"{config.output.execution_output.raw_selection}/{filename}")
        
    @staticmethod
    def _get_metrics(config: Config) -> List[Type[BaseStabilityMetric]]:
        metrics = set([stability.metric for stability in config.stability])
        metrics_class = []
        for metric in metrics:
            module = __import__('src.domain.stability.types.' + metric, fromlist=[metric])
            metric_class = getattr(module, metric)
            metrics_class.append(metric_class)
        return metrics_class