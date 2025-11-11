from enum import Enum


class SelectorType(Enum):
    RANK = 0
    WEIGHT = 1

_selection_compatibilities = {
    SelectorType.RANK: [SelectorType.RANK],
    SelectorType.WEIGHT: [SelectorType.RANK, SelectorType.WEIGHT]
}

def is_compatible_mode(_this: SelectorType, _with: SelectorType) -> bool:
    return _this in _selection_compatibilities[_with]