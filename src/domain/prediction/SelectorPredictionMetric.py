from typing import List
import numpy as np
from sklearn.metrics import classification_report

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
            report = SelectorPredictionMetric._calculate_classification_report(test_dataset, y_pred)
            score = SelectorPredictionScore(test_dataset.get_n_features(), selector, report)
        return score
    
    @staticmethod
    def _calculate_classification_report(dataset: Dataset, y_pred: np.ndarray) -> ClassificationScoreReport:
        y_true = dataset.get_labels()
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0.0)
        labels_score: List[ClassificationScoreLabelReport] = []

        accuracy = report['accuracy']
        weighted_avg_scores = report['weighted avg']
        precision = weighted_avg_scores['precision']
        recall = weighted_avg_scores['recall']
        f1_score = weighted_avg_scores['f1-score']
        support = weighted_avg_scores['support']
        general_score = ClassificationScoreGeneralReport(accuracy, precision, recall, f1_score, support)

        for label in range(0, dataset.get_n_labels()):
            label_score = report[str(label)]
            precision = label_score['precision']
            recall = label_score['recall']
            f1_score = label_score['f1-score']
            support = label_score['support']
            label_score = ClassificationScoreLabelReport(label, precision, recall, f1_score, support)
            labels_score.append(label_score)

        return ClassificationScoreReport(general_score, labels_score)
