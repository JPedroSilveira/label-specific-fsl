from typing import List

from src.domain.prediction.types.ClassificationScoreGeneralReport import ClassificationScoreGeneralReport
from src.domain.prediction.types.ClassificationScoreLabelReport import ClassificationScoreLabelReport


class ClassificationScoreReport:
    def __init__(self, general: ClassificationScoreGeneralReport, per_label: List[ClassificationScoreLabelReport]) -> None:
        self.general = general
        self.per_label = per_label