class ClassificationScoreAggregatedGeneralReport:
    def __init__(self, accuracy_avg: float, accuracy_stdev: float, precision_avg: float, precision_stdev: float, recall_avg: float, recall_stdev: float, f1_score_avg: float, f1_score_stdev: float) -> None:
        self.accuracy_avg = accuracy_avg
        self.accuracy_stdev = accuracy_stdev
        self.precision_avg = precision_avg
        self.precision_stdev = precision_stdev
        self.recall_avg = recall_avg
        self.recall_stdev = recall_stdev
        self.f1_score_avg = f1_score_avg
        self.f1_score_stdev = f1_score_stdev