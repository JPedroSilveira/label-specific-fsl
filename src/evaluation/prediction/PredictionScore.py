from src.domain.predictor.types.base.BasePredictor import BasePredictor
from src.domain.selector.types.base.BaseSelector import BaseSelector
from src.util.classification_report_util import ClassificationScoreReport


class BasePredictionScore():
    def __init__(self, selector: BaseSelector, selection_size: int, report: ClassificationScoreReport) -> None:
        self.selector_name = selector.get_class_name()
        self.report: ClassificationScoreReport = report
        self.selection_size = selection_size

class SelectorPredictionScore(BasePredictionScore):
    def __init__(self, selection_size: int, selector: BaseSelector, report: ClassificationScoreReport) -> None:
        super().__init__(selector, selection_size, report)

class PredictorPredictionScore(BasePredictionScore):
    def __init__(self, selection_size: int, selector: BaseSelector, predictor: BasePredictor, evaluator_name: str, report: ClassificationScoreReport) -> None:
        super().__init__(selector, selection_size, report)
        self.predictor_name = predictor.get_class_name()
        self.evaluator_name = evaluator_name