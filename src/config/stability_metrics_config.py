from typing import Type, List
from src.evaluation.stability.metric.BaseStabilityMetric import BaseStabilityMetric
from src.evaluation.stability.metric.JaccardMetric import JaccardMetric
from src.evaluation.stability.metric.SpearmanMetric import SpearmanMetric
from src.evaluation.stability.metric.PearsonMetric import PearsonMetric
from src.evaluation.stability.metric.KunchevaMetric import KunchevaMetric


STABILITY_METRICS_TYPES: List[Type[BaseStabilityMetric]] = [JaccardMetric, SpearmanMetric, PearsonMetric]