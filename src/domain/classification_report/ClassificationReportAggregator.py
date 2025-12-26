import statistics
from typing import Dict, List

from src.domain.prediction.types.ClassificationScoreAggregatedReport import ClassificationScoreAggregatedReport
from src.domain.prediction.types.ClassificationScoreAggregatedGeneralReport import ClassificationScoreAggregatedGeneralReport
from src.domain.prediction.types.ClassificationScoreAggregatedPerLabelReport import ClassificationScoreAggregatedPerLabelReport
from src.domain.prediction.types.ClassificationScoreLabelReport import ClassificationScoreLabelReport
from src.domain.prediction.types.ClassificationScoreGeneralReport import ClassificationScoreGeneralReport
from src.domain.log.Logger import Logger


class ClassificationReportAggregator:
    @classmethod
    def execute(cls, general_reports: List[ClassificationScoreGeneralReport], per_label_reports: Dict[int, List[ClassificationScoreLabelReport]]) -> ClassificationScoreAggregatedReport:
        general_aggregated_report: ClassificationScoreAggregatedGeneralReport = None
        per_label_aggregated_reports: List[ClassificationScoreAggregatedPerLabelReport] = []
        if general_reports is not None:
            Logger.execute(f"-- Label: general")
            general_aggregated_report = cls._aggregate_general_reports(general_reports)
        else:
            Logger.execute(f"-- General Prediction Reports: No entries found")
        if per_label_reports is not None:
            for label, label_prediction_reports in per_label_reports.items():
                Logger.execute(f"-- Label {label}:")
                per_label_aggregated_report = cls._aggregate_per_label_reports(label_prediction_reports)
                per_label_aggregated_reports.append(per_label_aggregated_report)
        else:
            Logger.execute(f"-- Per Label Prediction Reports: No entries found")
        return ClassificationScoreAggregatedReport(
            general=general_aggregated_report,
            per_label=per_label_aggregated_reports
        )
            
    @staticmethod
    def _aggregate_general_reports(reports: List[ClassificationScoreGeneralReport]) -> ClassificationScoreAggregatedGeneralReport:
        if len(reports) == 0:
            Logger.execute(f"--- No entries found")
            return
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
        return ClassificationScoreAggregatedGeneralReport(
            accuracy_avg=avg_accuracy,
            accuracy_stdev=stdev_accuracy,
            precision_avg=avg_precision,
            precision_stdev=stdev_precision,
            recall_avg=avg_recall,
            recall_stdev=stdev_recall,
            f1_score_avg=avg_f1_score,
            f1_score_stdev=stdev_f1_score
        )
            
    @staticmethod
    def _aggregate_per_label_reports(reports: List[ClassificationScoreLabelReport]) -> ClassificationScoreAggregatedPerLabelReport:
        if len(reports) == 0:
            Logger.execute(f"--- No entries found")
            return
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
        return ClassificationScoreAggregatedPerLabelReport(
            label=reports[0].label,
            precision_avg=avg_precision,
            precision_stdev=stdev_precision,
            recall_avg=avg_recall,
            recall_stdev=stdev_recall,
            f1_score_avg=avg_f1_score,
            f1_score_stdev=stdev_f1_score
        )