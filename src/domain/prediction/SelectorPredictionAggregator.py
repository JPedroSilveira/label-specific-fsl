import statistics
from typing import List, Type

from fontTools.merge.util import first

from src.domain.classification_report.ClassificationReportAggregator import ClassificationReportAggregator
from src.domain.prediction.types.ClassificationScoreGeneralReport import ClassificationScoreGeneralReport
from src.domain.prediction.types.ClassificationScoreLabelReport import ClassificationScoreLabelReport
from src.domain.log.Logger import Logger
from src.domain.selector.types.base.BaseSelector import BaseSelector
from src.domain.storage.ExecutionStorage import ExecutionStorage


class SelectorPredictionAggregator:
    @staticmethod
    def execute(selectors: List[Type[BaseSelector]], storage: ExecutionStorage) -> None:
        Logger.execute("Metric: Selector Prediction")
        for selector_class in selectors:
            Logger.execute(f"- Selector: {selector_class.get_name()}")
            general_prediction_reports = storage.get_general_prediction_score(selector_class)
            per_label_prediction_reports = storage.get_per_label_prediction_scores(selector_class)
            ClassificationReportAggregator.execute(general_prediction_reports, per_label_prediction_reports)