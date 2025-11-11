from src.domain.selector.types.base.BaseSelector import BaseSelector
from src.util.classification_report_util import ClassificationScoreReport


class OcclusionScore:
    def __init__(self, selector: BaseSelector, removed_features: int, report: ClassificationScoreReport, loss: float, loss_by_label: dict[int, float], inverse_report: ClassificationScoreReport) -> None:
        self.report: ClassificationScoreReport = report
        self.loss = loss
        self.loss_by_label = loss_by_label
        self.removed_features = removed_features
        self.selector_name = selector.get_selector_name()
        self.inverse_report: ClassificationScoreReport = inverse_report

    def __str__(self) -> str:
        output = f"\n========================================\nSelector: {self.selector_name}\nSelection Size: {self.removed_features}\n\nLoss: {self.loss}\n\nAcurracy: {self.report.general.accuracy}\n\nWeighted Average Scores:\n  Precision: {self.report.general.precision}\n  Recall: {self.report.general.recall}\n  F1-Score: {self.report.general.f1_score}\n  Support: {self.report.general.support}\n\nBy Label Scores:\n"
        for label, score in enumerate(self.report.per_label):
            output += f"  Average Scores (Label {label}):\n    Loss: {self.loss_by_label[label]}\n    Precision: {score.precision}\n    Recall: {score.recall}\n    F1-Score: {score.f1_score}\n    Support: {score.support}\n"
        output += "========================================\n"
        return output

class OcclusionScorePerLabel(OcclusionScore):
    def __init__(self, selector: BaseSelector, removed_features: int, label: int, report: ClassificationScoreReport, loss: float, loss_by_label: dict[int, float], inverse_report: ClassificationScoreReport) -> None:
        super().__init__(selector, removed_features, report, loss, loss_by_label, inverse_report)
        self.label = label

    def __str__(self) -> str:
        output = f"\n========================================\nSelector: {self.selector_name}\nEvaluated Label: {self.label}\nSelection Size: {self.removed_features}\n\nLoss: {self.loss}\n\nAcurracy: {self.report.general.accuracy}\n\nWeighted Average Scores:\n  Precision: {self.report.general.precision}\n  Recall: {self.report.general.recall}\n  F1-Score: {self.report.general.f1_score}\n  Support: {self.report.general.support}\n\nBy Label Scores:\n"
        for label, score in enumerate(self.report.per_label):
            output += f"  Average Scores (Label {label}):\n    Loss: {self.loss_by_label[label]}\n    Precision: {score.precision}\n    Recall: {score.recall}\n    F1-Score: {score.f1_score}\n    Support: {score.support}\n"
        output += "========================================\n"
        return output