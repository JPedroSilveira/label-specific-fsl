from src.domain.classification_report.ClassificationReportCalculator import ClassificationReportCalculator
from src.evaluation.prediction.PredictionScore import SelectorPredictionScore
from src.domain.selector.types.base.BaseSelector import BaseSelector
from src.domain.storage.ExecutionStorage import ExecutionStorage
from src.domain.data.types.Dataset import Dataset
from src.domain.log.Logger import Logger


class SelectorPredictionMetric:
    @staticmethod
    def execute(selector: BaseSelector, test_dataset: Dataset, storage: ExecutionStorage) -> None:
        Logger.execute("Metric: Selector Prediction")
        score = None
        if not selector.can_predict():
            Logger.execute(f"- Selector can not do predictions, skipping...")
        else:
            y_pred = selector.predict(test_dataset)
            report = ClassificationReportCalculator.execute(test_dataset.get_labels(), y_pred, test_dataset.get_n_labels())
            score = SelectorPredictionScore(test_dataset.get_n_features(), selector, report)
            storage.add_general_prediction_score(selector, score.report.general)
            for label, label_score in enumerate(score.report.per_label):
                storage.add_per_label_prediction_score(selector, label, label_score)
            Logger.execute(f'F1 Score: {score.report.general.f1_score}')