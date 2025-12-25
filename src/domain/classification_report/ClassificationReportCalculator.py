from typing import List
import numpy as np
from sklearn.metrics import classification_report

from src.domain.prediction.types.ClassificationScoreLabelReport import ClassificationScoreLabelReport
from src.domain.prediction.types.ClassificationScoreGeneralReport import ClassificationScoreGeneralReport
from src.domain.prediction.types.ClassificationScoreReport import ClassificationScoreReport


class ClassificationReportCalculator:
    @staticmethod
    def execute(y_true: np.ndarray, y_pred: np.ndarray, n_labels: int) -> ClassificationScoreReport:
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0.0)
        labels_score: List[ClassificationScoreLabelReport] = []
        accuracy = report['accuracy']
        weighted_avg_scores = report['weighted avg']
        precision = weighted_avg_scores['precision']
        recall = weighted_avg_scores['recall']
        f1_score = weighted_avg_scores['f1-score']
        support = weighted_avg_scores['support']
        general_score = ClassificationScoreGeneralReport(accuracy, precision, recall, f1_score, support)
        for label in range(0, n_labels):
            label_score = report[str(label)]
            precision = label_score['precision']
            recall = label_score['recall']
            f1_score = label_score['f1-score']
            support = label_score['support']
            label_score = ClassificationScoreLabelReport(label, precision, recall, f1_score, support)
            labels_score.append(label_score)
        return ClassificationScoreReport(general_score, labels_score)