class ClassificationScoreLabelReport:
    def __init__(self, label: int, precision: float, recall: float, f1_score: float, support: int) -> None:
        self.label = label
        self.precision = precision
        self.recall = recall
        self.f1_score = f1_score
        self.support = support