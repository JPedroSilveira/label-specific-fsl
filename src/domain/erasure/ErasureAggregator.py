from typing import Dict, List, Type

from config.type import Config
from src.domain.log.Logger import Logger
from src.domain.storage.ExecutionStorage import ExecutionStorage
from src.domain.informative_features.InformativeFeaturesCalculator import BaseSelector
from src.domain.classification_report.ClassificationReportAggregator import ClassificationReportAggregator
from src.domain.prediction.types.ClassificationScoreAggregatedReport import ClassificationScoreAggregatedReport
from src.domain.classification_report.ClassificationReportAggregatedGraph import ClassificationReportAggregatedGraph


class ErasureAggregator:
    @staticmethod
    def execute(selectors_class: List[Type[BaseSelector]], storage: ExecutionStorage, config: Config) -> None:
        Logger.execute("Metric: Erasure Evaluation Aggregation")
        for selector_class in selectors_class:
            Logger.execute(f"- Selector: {selector_class.get_name()}")
            erasure_scores_per_label_and_number_of_features = storage.get_erasure_scores_per_selector_and_label_and_number_of_features(selector_class)
            for label, erasure_scores_per_number_of_features in erasure_scores_per_label_and_number_of_features.items():
                Logger.execute(f"-> Ranking from label: {label}")
                aggregated_report_per_number_of_features: Dict[int, ClassificationScoreAggregatedReport] = {}
                for number_of_features, erasure_scores in erasure_scores_per_number_of_features.items():
                    Logger.execute(f"-> Number of features: {number_of_features}")
                    general_erasure_scores, per_label_erasure_scores = erasure_scores
                    aggregated_report = ClassificationReportAggregator.execute(general_erasure_scores, per_label_erasure_scores)
                    aggregated_report_per_number_of_features[number_of_features] = aggregated_report
                ClassificationReportAggregatedGraph.execute(selector_class.get_name(), label, aggregated_report_per_number_of_features, config.output.execution_output.feature_erasure)
                    