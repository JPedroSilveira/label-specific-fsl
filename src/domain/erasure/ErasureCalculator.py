import numpy as np
from typing import Dict
from config.type import Config, List

from src.domain.log.Logger import Logger
from src.domain.data.types.Dataset import Dataset
from src.domain.storage.ExecutionStorage import ExecutionStorage
from src.domain.selector.types.enum.SelectorSpecificity import SelectorSpecificity
from src.domain.informative_features.InformativeFeaturesCalculator import BaseSelector
from src.domain.prediction.types.ClassificationScoreLabelReport import ClassificationScoreLabelReport
from src.domain.prediction.types.ClassificationScoreGeneralReport import ClassificationScoreGeneralReport
from src.domain.classification_report.ClassificationReportCalculator import ClassificationReportCalculator


class ErasureCalculator:
    @classmethod
    def execute(cls, selector: BaseSelector, test_dataset: Dataset, storage: ExecutionStorage, config: Config) -> None:
        Logger.execute("Metric: Erasure Evaluation Calculation")
        if not selector.can_predict():
            Logger.execute(f"- Selector can not do predictions, skipping...")
        elif selector.get_specificity() == SelectorSpecificity.PER_LABEL:
            Logger.execute(f"- Calculating general and per label metric...")
            cls._calculate_erasure_metric(selector, test_dataset, storage, config, "general", selector.get_general_ranking())
            for label, label_ranking in enumerate(selector.get_per_label_ranking()):
                cls._calculate_erasure_metric(selector, test_dataset, storage, config, str(label), label_ranking)
        elif selector.get_specificity() == SelectorSpecificity.GENERAL:
            Logger.execute(f"- Calculating general metric only...")
            cls._calculate_erasure_metric(selector, test_dataset, storage, config, "general", selector.get_general_ranking())

    @staticmethod
    def _calculate_erasure_metric(selector: BaseSelector, test_dataset: Dataset, storage: ExecutionStorage, config: Config, label: str, ranking: np.ndarray) -> None:
        X_test = test_dataset.get_features()
        y_test = test_dataset.get_labels()
        for i in range(0, config.dataset.erasure_k + 1, config.dataset.erasure_step):
            number_of_features = i
            Logger.execute(f"-- Number of features removed: {number_of_features}")
            selected_features = ranking[0:number_of_features]
            X_test_selected = X_test.copy()
            X_test_selected[:, selected_features] = 0
            y_pred = selector.predict(Dataset.from_dataset_with_new_features(test_dataset, X_test_selected))
            report = ClassificationReportCalculator.execute(y_test, y_pred, test_dataset.get_n_labels())
            storage.add_erasure_scores_per_selector_and_label_and_number_of_features(selector, label, number_of_features, report.general, report.per_label)       
            Logger.execute(f"--- F1 Score: {report.general.f1_score}")
