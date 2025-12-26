class ClassificationScoreAggregatedPerLabelReport:
    def __init__(self, label: int, precision_avg: float, precision_stdev: float, recall_avg: float, recall_stdev, f1_score_avg: float, f1_score_stdev: float) -> None:
        self.label = label
        self.precision_avg = precision_avg
        self.precision_stdev = precision_stdev
        self.recall_avg = recall_avg
        self.recall_stdev = recall_stdev
        self.f1_score_avg = f1_score_avg
        self.f1_score_stdev = f1_score_stdev