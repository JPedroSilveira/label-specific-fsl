import statistics
from typing import List, Type

from fontTools.merge.util import first

from src.domain.prediction.types.ClassificationScoreGeneralReport import ClassificationScoreGeneralReport
from src.domain.prediction.types.ClassificationScoreLabelReport import ClassificationScoreLabelReport
from src.domain.log.Logger import Logger
from src.domain.selector.types.base.BaseSelector import BaseSelector
from src.domain.storage.ExecutionStorage import ExecutionStorage


class SelectorPredictionAggregator:
    @classmethod
    def execute(cls, selectors: List[Type[BaseSelector]], storage: ExecutionStorage) -> None:
        Logger.execute("Metric: Selector Prediction")
        for selector_class in selectors:
            Logger.execute(f"- Selector: {selector_class.get_name()}")
            general_prediction_reports = storage.get_general_prediction_score(selector_class)
            per_label_prediction_reports = storage.get_per_label_prediction_scores(selector_class)
            if general_prediction_reports is not None:
                Logger.execute(f"-- Label: general")
                cls._aggregate_prediction_reports(general_prediction_reports)
            else:
                Logger.execute(f"-- General Prediction Reports: No entries found")
            if per_label_prediction_reports is not None:
                for label, label_prediction_reports in per_label_prediction_reports.items():
                    Logger.execute(f"-- Label {label}:")
                    cls._aggregate_prediction_reports(label_prediction_reports)
            else:
                Logger.execute(f"-- Per Label Prediction Reports: No entries found")
        
    def _aggregate_prediction_reports(reports: List[ClassificationScoreGeneralReport | ClassificationScoreLabelReport]) -> None:
        if len(reports) == 0:
            Logger.execute(f"--- No entries found")
            return
        first_report = first(reports)
        if isinstance(first_report, ClassificationScoreGeneralReport):
            avg_accuracy = statistics.mean([report.accuracy for report in reports])
            avg_precision = statistics.mean([report.precision for report in reports])
            avg_recall = statistics.mean([report.recall for report in reports])
            avg_f1_score = statistics.mean([report.f1_score for report in reports])
            avg_support = statistics.mean([report.support for report in reports])
            stdev_accuracy = statistics.stdev([report.accuracy for report in reports])
            stdev_precision = statistics.stdev([report.precision for report in reports])
            stdev_recall = statistics.stdev([report.recall for report in reports])
            stdev_f1_score = statistics.stdev([report.f1_score for report in reports])
            stdev_support = statistics.stdev([report.support for report in reports])
            Logger.execute(f"--- Aggregated Report: Accuracy={avg_accuracy} (±{stdev_accuracy}), Precision={avg_precision} (±{stdev_precision}), Recall={avg_recall} (±{stdev_recall}), F1 Score={avg_f1_score} (±{stdev_f1_score}), Support={avg_support} (±{stdev_support})")
            Logger.execute(f"--- Accuracy List: {[report.accuracy for report in reports]}")
            Logger.execute(f"--- Precision List: {[report.precision for report in reports]}")
            Logger.execute(f"--- Recall List: {[report.recall for report in reports]}")
            Logger.execute(f"--- F1 Score List: {[report.f1_score for report in reports]}")
            Logger.execute(f"--- Support List: {[report.support for report in reports]}")
        elif isinstance(first_report, ClassificationScoreLabelReport):
            avg_precision = statistics.mean([report.precision for report in reports])
            avg_recall = statistics.mean([report.recall for report in reports])
            avg_f1_score = statistics.mean([report.f1_score for report in reports])
            avg_support = statistics.mean([report.support for report in reports])
            stdev_precision = statistics.stdev([report.precision for report in reports])
            stdev_recall = statistics.stdev([report.recall for report in reports])
            stdev_f1_score = statistics.stdev([report.f1_score for report in reports])
            stdev_support = statistics.stdev([report.support for report in reports])
            Logger.execute(f"--- Aggregated Report: Precision={avg_precision} (±{stdev_precision}), Recall={avg_recall} (±{stdev_recall}), F1 Score={avg_f1_score} (±{stdev_f1_score}), Support={avg_support} (±{stdev_support})")
            Logger.execute(f"--- Precision List: {[report.precision for report in reports]}")
            Logger.execute(f"--- Recall List: {[report.recall for report in reports]}")
            Logger.execute(f"--- F1 Score List: {[report.f1_score for report in reports]}")
            Logger.execute(f"--- Support List: {[report.support for report in reports]}")
