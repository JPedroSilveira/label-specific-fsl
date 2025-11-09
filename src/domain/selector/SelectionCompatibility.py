from model.SelectorType import SelectorType


_selection_compatibilities = {
    SelectorType.RANK: [SelectorType.RANK],
    SelectorType.WEIGHT: [SelectorType.RANK, SelectorType.WEIGHT]
}

class SelectionCompatibility:
    @staticmethod
    def execute(_this: SelectorType, _with: SelectorType) -> bool:
        return _this in _selection_compatibilities[_with]