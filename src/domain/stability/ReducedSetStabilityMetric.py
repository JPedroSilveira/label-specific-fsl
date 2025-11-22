import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Type

from config.type import Config
from src.domain.selector.types.enum.SelectorSpecificity import SelectorSpecificity
from src.domain.stability.StabilityMetricTypeCreator import StabilityMetricTypeCreator
from src.domain.stability.types.base.BaseStabilityMetric import BaseStabilityMetric
from src.domain.data.FeatureRemover import FeatureRemover
from src.domain.data.types.Dataset import Dataset
from src.domain.selector.types.base.BaseSelector import BaseSelector


class ReducedSetStabilityMetric:
    @classmethod
    def execute(cls, selector: BaseSelector, selector_class: Type[BaseSelector], train_dataset: Dataset, test_dataset: Dataset, config: Config) -> Dict[str, Dict[str, Dict[str, float]]]:
        score_per_size_per_label_per_metric: Dict[str, Dict[str, Dict[str, float]]] = {}
        metrics = StabilityMetricTypeCreator.execute(config)
        for size in config.stability_reduced:
            reduced_train_dataset, top_k_features = cls._reduce_dataset(selector, train_dataset, size)
            reduced_test_dataset, _ = cls._reduce_dataset(selector, test_dataset, size)
            reduced_selector = selector_class(reduced_train_dataset.get_n_features(), reduced_train_dataset.get_n_labels(), config.dataset)
            reduced_selector.fit(reduced_train_dataset, reduced_test_dataset)
            general_score_per_metric = cls._compare_general_weights(metrics, selector, reduced_selector, train_dataset, reduced_train_dataset, top_k_features, config)
            score_per_label_per_metric = cls._compare_per_label_weights(metrics, selector, reduced_selector, train_dataset, reduced_train_dataset, top_k_features, config)
            score_per_label_per_metric["General"] = general_score_per_metric
            score_per_size_per_label_per_metric[size] = score_per_label_per_metric
        return score_per_size_per_label_per_metric
            
    @classmethod
    def _compare_general_weights(cls, metrics: List[Type[BaseStabilityMetric]], selector: BaseSelector, reduced_selector: BaseSelector, train_dataset: Dataset, reduced_train_dataset: Dataset, top_k_features: np.ndarray, config: Config) -> Dict[str, float]:
        selector_weights_df = cls._get_reduced_weights(selector.get_general_weights(), train_dataset.get_feature_names(), top_k_features)
        reduced_selector_weight_df = cls._get_weights_as_dataframe(reduced_selector.get_general_weights(), reduced_train_dataset.get_feature_names())
        score_per_metric: Dict[str,float] = {}
        for metric in metrics:
            score = metric.execute(selector_weights_df, reduced_selector_weight_df, config)
            score_per_metric[metric.get_name()] = score
        return score_per_metric
    
    @classmethod
    def _compare_per_label_weights(cls, metrics: List[Type[BaseStabilityMetric]], selector: BaseSelector, reduced_selector: BaseSelector, train_dataset: Dataset, reduced_train_dataset: Dataset, top_k_features: np.ndarray, config: Config) -> Dict[str, Dict[str, float]]:
        score_per_label_per_metric: Dict[str, Dict[str, float]] = {}
        if selector.get_specificity() is SelectorSpecificity.PER_LABEL:
            selector_per_label_weights = selector.get_per_label_weights()
            reduced_selector_per_label_weights = reduced_selector.get_per_label_weights()
            if selector_per_label_weights is not None:
                for label, selector_weights in enumerate(selector_per_label_weights):
                    reduced_selector_weights = reduced_selector_per_label_weights[label]
                    selector_weights_df = cls._get_reduced_weights(selector_weights, train_dataset.get_feature_names(), top_k_features)
                    reduced_selector_weight_df = cls._get_weights_as_dataframe(reduced_selector_weights, reduced_train_dataset.get_feature_names())
                    score_per_metric: Dict[str,float] = {}
                    for metric in metrics:
                        score = metric.execute(selector_weights_df, reduced_selector_weight_df, config)
                        score_per_metric[metric.get_name()] = score
                    score_per_label_per_metric[label] = score_per_metric
        return score_per_label_per_metric
            
    @classmethod
    def _get_reduced_weights(cls, weights: np.ndarray, feature_names: List[str], top_k_features: np.ndarray) -> np.ndarray:
        dataframe = cls._get_weights_as_dataframe(weights, feature_names)
        condition = ~dataframe['feature'].isin(top_k_features)
        return dataframe[condition].reset_index(drop=True)
    
    @staticmethod
    def _get_weights_as_dataframe(weights: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
        return pd.DataFrame(
            list(zip(feature_names, weights)), 
            columns=["feature", "value"]
        )
        
    @staticmethod
    def _reduce_dataset(selector: BaseSelector, dataset: Dataset, size: int) -> Tuple[Dataset, np.ndarray]:
        weights = selector.get_general_weights()
        dataframe = pd.DataFrame(
            list(zip(dataset.get_feature_names(), weights)), 
            columns=["feature", "value"]
        )
        top_k_dataframe = dataframe.nlargest(size, 'value')
        top_k_features = top_k_dataframe['feature']
        new_dataset = dataset
        for feature_name in top_k_features:
            new_dataset = FeatureRemover.execute(new_dataset, feature_name)
        return new_dataset, top_k_features.to_numpy()
        
    @staticmethod
    def _get_metrics(config: Config) -> List[Type[BaseStabilityMetric]]:
        metrics = set([stability.name for stability in config.stability_metric])
        metrics_class = []
        for metric in metrics:
            module = __import__('src.domain.stability.types.' + metric, fromlist=[metric])
            metric_class = getattr(module, metric)
            metrics_class.append(metric_class)
        return metrics_class