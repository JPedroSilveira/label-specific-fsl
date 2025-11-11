class ClassificationScoreGeneralReport:
    def __init__(self, accuracy: float, precision: float, recall: float, f1_score: float, support: int) -> None:
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1_score = f1_score
        self.support = support