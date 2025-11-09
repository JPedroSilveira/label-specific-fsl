import numpy as np

from src.model.SelectorType import SelectorType
from src.domain.selector.types.base.BaseSelector import BaseSelector


class BaseSelectorRank(BaseSelector):
    def __init__(self, n_features, n_labels) -> None:
        super().__init__(n_features, n_labels)

    def get_type(self) -> SelectorType:
        return SelectorType.RANK