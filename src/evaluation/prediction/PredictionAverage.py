class PredictionScoreAverageByLabel():
    def __init__(self) -> None:
        self.label = None
        self.precision_avg = None
        self.precision_var = None
        self.recall_avg = None
        self.recall_var = None
        self.f1_score_avg = None
        self.f1_score_var = None
        self.support_avg = None
        self.support_var = None
        
class BasePredictionScoreStatistics():
    def __init__(self) -> None:
        self.accuracy_avg = None
        self.accuracy_var = None
        self.precision_avg = None
        self.precision_var = None
        self.recall_avg = None
        self.recall_var = None
        self.f1_score_avg = None
        self.f1_score_var = None
        self.support_avg = None
        self.support_var = None
        self.by_label: list[PredictionScoreAverageByLabel] = []
        self.selector_name = None

class SelectorPredictionScoreStatistics(BasePredictionScoreStatistics):
    def __init__(self) -> None:
        super().__init__()
    
class PredictorPredictionScoreStatistics(BasePredictionScoreStatistics):
    def __init__(self) -> None:
        super().__init__()
        self.predictor_name = None
        self.evaluator_name = None
        self.selection_size = None

    