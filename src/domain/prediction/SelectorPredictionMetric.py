from typing import List
import numpy as np
from sklearn.metrics import classification_report
from sympy.codegen.cnodes import static

from src.domain.classification_report.ClassificationReportCalculator import ClassificationReportCalculator
from src.domain.prediction.types.ClassificationScoreLabelReport import ClassificationScoreLabelReport
from src.domain.prediction.types.ClassificationScoreGeneralReport import ClassificationScoreGeneralReport
from src.domain.prediction.types.ClassificationScoreReport import ClassificationScoreReport
from src.evaluation.prediction.PredictionScore import SelectorPredictionScore
from src.domain.data.types.Dataset import Dataset
from src.domain.selector.types.base.BaseSelector import BaseSelector


class SelectorPredictionMetric:
    @staticmethod
    def execute(selector: BaseSelector, test_dataset: Dataset) -> SelectorPredictionScore:
        score = None
        if selector.can_predict():
            y_pred = selector.predict(test_dataset)
            report = ClassificationReportCalculator.execute(test_dataset.get_labels(), y_pred, test_dataset.get_n_labels())
            score = SelectorPredictionScore(test_dataset.get_n_features(), selector, report)
        return score