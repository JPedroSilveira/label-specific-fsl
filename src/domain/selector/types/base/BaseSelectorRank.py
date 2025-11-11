import numpy as np

from config.type import DatasetConfig
from src.domain.selector.types.enum.SelectorType import SelectorType
from src.domain.selector.types.base.BaseSelector import BaseSelector


class BaseSelectorRank(BaseSelector):
    def __init__(self, n_features: int, n_labels: int, config: DatasetConfig) -> None:
        super().__init__(n_features, n_labels, config)

    def get_type(self) -> SelectorType:
        return SelectorType.RANK