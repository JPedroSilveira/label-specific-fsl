from typing import List
import numpy as np
from src.model.Dataset import Dataset
from sklearn.metrics import classification_report


class ClassificationScoreGeneralReport:
    def __init__(self, accuracy: float, precision: float, recall: float, f1_score: float, support: int) -> None:
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1_score = f1_score
        self.support = support

class ClassificationScoreLabelReport:
    def __init__(self, label: int, precision: float, recall: float, f1_score: float, support: int) -> None:
        self.label = label
        self.precision = precision
        self.recall = recall
        self.f1_score = f1_score
        self.support = support

class ClassificationScoreReport():
    def __init__(self, general: ClassificationScoreGeneralReport, per_label: List[ClassificationScoreLabelReport]) -> None:
        self.general = general
        self.per_label = per_label

def calculate_classification_report(dataset: Dataset, y_pred: np.ndarray):
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

    