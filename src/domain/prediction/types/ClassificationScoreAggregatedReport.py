from typing import List

from src.domain.prediction.types.ClassificationScoreAggregatedGeneralReport import ClassificationScoreAggregatedGeneralReport
from src.domain.prediction.types.ClassificationScoreAggregatedPerLabelReport import ClassificationScoreAggregatedPerLabelReport


class ClassificationScoreAggregatedReport:
    def __init__(self, general: ClassificationScoreAggregatedGeneralReport, per_label: List[ClassificationScoreAggregatedPerLabelReport]) -> None:
        self.general = general
        self.per_label = per_label