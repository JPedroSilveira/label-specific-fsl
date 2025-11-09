class InformativeFeaturesScore():
    def __init__(self, selector_name: str, selection_size: int, percentage_of_informative_features_selected: int, percentage_of_selected_features_that_are_informative: int, label: str = "General") -> None:
        self._selector_name = selector_name
        self._selection_size = selection_size
        self._percentage_of_informative_features_selected = percentage_of_informative_features_selected
        self._percentage_of_selected_features_that_are_informative = percentage_of_selected_features_that_are_informative
        self._label = label

    def get_selector_name(self):
        return self._selector_name

    def get_selection_size(self):
        return self._selection_size
    
    def get_percentage_of_informative_features_selected(self):
        return self._percentage_of_informative_features_selected
    
    def get_percentage_of_selected_features_that_are_informative(self):
        return self._percentage_of_selected_features_that_are_informative
    
    def get_label(self):
        return self._label

    def __str__(self) -> str:
        return f"Selector: {self._selector_name} \nSelection size: {self._selection_size} \nLabel: {self._label} \nPercentage of informative features selected: {self._percentage_of_informative_features_selected * 100}% \nPercentagem of selected features that are informative: {self._percentage_of_selected_features_that_are_informative * 100}%"