import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from typing import Any, Dict, List, Type

from config.type import Config
from src.domain.log.Logger import Logger
from src.domain.data.reader.RankingReader import RankingReader
from src.domain.data.types.SplittedDataset import SplittedDataset
from src.domain.informative_features.InformativeFeaturesCalculator import BaseSelector
from src.domain.prediction.types.ClassificationScoreLabelReport import ClassificationScoreLabelReport
from src.domain.prediction.types.ClassificationScoreGeneralReport import ClassificationScoreGeneralReport
from src.domain.classification_report.ClassificationReportCalculator import ClassificationReportCalculator
from src.domain.classification_report.ClassificationReportAggregator import ClassificationReportAggregator
from src.domain.prediction.types.ClassificationScoreAggregatedReport import ClassificationScoreAggregatedReport
from src.domain.classification_report.ClassificationReportAggregatedGraph import ClassificationReportAggregatedGraph


class ExternalPredictorEvaluation:
    @classmethod
    def execute(cls, selectors_class: List[Type[BaseSelector]], splitted_dataset: SplittedDataset, config: Config) -> None:
        Logger.execute("Metric: External Predictor Evaluation")
        for selector_class in selectors_class:
            Logger.execute(f"- Selector: {selector_class.get_name()}")
            ranking_per_specificity = RankingReader.execute(selector_class, config)
            for specificity in ranking_per_specificity.keys():
                ranking_list = ranking_per_specificity[specificity]
                if specificity == "general":
                    Logger.execute(f"-> Ranking by label: {specificity}")
                    cls._evaluate_selected_features(selector_class.get_name(), specificity, splitted_dataset, ranking_list, config)
                elif specificity.startswith("label"):
                    label = int(specificity.replace('label', ''))
                    Logger.execute(f"-> Ranking by label: {label}")
                    cls._evaluate_selected_features(selector_class.get_name(), label, splitted_dataset, ranking_list, config)
                else:
                    Logger.execute(f"-- [ERROR] Specificity {specificity} is invalid!")
            
    @classmethod
    def _evaluate_selected_features(cls, selector_name: str, ranking_label: str, splitted_dataset: SplittedDataset, ranking_list: List[pd.DataFrame], config: Config) -> None:
        feature_names = splitted_dataset.get_feature_names()
        X_train = splitted_dataset.get_train().get_features()
        y_train = splitted_dataset.get_train().get_labels()
        X_test = splitted_dataset.get_test().get_features()
        y_test = splitted_dataset.get_test().get_labels()
        aggregated_report_per_number_of_features: Dict[int, ClassificationScoreAggregatedReport] = {}
        for i in range(0, config.dataset.external_predictor_k + 1, config.dataset.external_predictor_step):
            number_of_features = i
            Logger.execute(f"-> Number of features removed: {number_of_features}")
            general_reports: List[ClassificationScoreGeneralReport] = []
            per_label_reports: Dict[int, List[ClassificationScoreLabelReport]] = {}
            for ranking in ranking_list:
                selected_features = ranking['feature'].to_numpy()[0:number_of_features]
                selected_features_index = [feature_names.index(feature_name) for feature_name in selected_features]
                X_train_selected = np.delete(X_train, selected_features_index, axis=1)
                X_test_selected = np.delete(X_test, selected_features_index, axis=1)
                svc_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=config.dataset.random_seed)
                svc_model.fit(X_train_selected, y_train)
                y_pred = svc_model.predict(X_test_selected)
                report = ClassificationReportCalculator.execute(y_test, y_pred, splitted_dataset.get_n_labels())
                general_reports.append(report.general)
                for label_report in report.per_label:
                    if label_report.label not in per_label_reports:
                        per_label_reports[label_report.label] = []
                    per_label_reports[label_report.label].append(label_report)
            aggregated_report = ClassificationReportAggregator.execute(general_reports, per_label_reports)
            aggregated_report_per_number_of_features[number_of_features] = aggregated_report
        ClassificationReportAggregatedGraph.execute(selector_name, ranking_label, aggregated_report_per_number_of_features, config.output.execution_output.selection_performance)