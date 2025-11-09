from enum import Enum


class SelectionMode(Enum):
    RANK = 0
    WEIGHT = 1

selection_compatibilities = {
    SelectionMode.RANK: [SelectionMode.RANK],
    SelectionMode.WEIGHT: [SelectionMode.RANK, SelectionMode.WEIGHT]
}


def is_compatible_mode(_this: SelectionMode, _with: SelectionMode):
    return _this in selection_compatibilities[_with]