import numpy as np
from src.model.Dataset import Dataset
from src.selector.enum.SelectionMode import SelectionMode
from src.selector.enum.PredictionMode import PredictionMode
from src.selector.enum.SelectionSpecificity import SelectionSpecificity



class BaseSelectorWrapper():
    def __init__(self, n_features:int, n_labels:int) -> None:
        self._n_labels = n_labels
        self._n_features = n_features
        pass

    def get_name() -> str:
        raise NotImplemented("")

    def get_class_name(self) -> str:
        return self.__class__.get_name()
    
    def get_n_labels(self) -> int:
        return self._n_labels
    
    def get_n_features(self) -> int:
        return self._n_features

    def fit(self, train_dataset: Dataset, test_dataset: Dataset) -> None:
        raise NotImplemented("")
    
    def predict(self, dataset: Dataset) -> np.ndarray:
        raise NotImplemented("")
    
    def predict_probabilities(self, dataset: Dataset, use_softmax: bool=True) -> np.ndarray:
        raise NotImplemented("")
    
    def get_prediction_mode(self) -> PredictionMode:
        raise NotImplemented("")
    
    def get_selection_mode(self) -> SelectionMode: 
        raise NotImplemented("")
    
    def get_selection_specificities(self) -> list[SelectionSpecificity]:
        raise NotImplemented("")
    
    def get_general_weights(self) -> np.ndarray:
        """
            Should be implemented when SelectionMode = WEIGHT and GENERAL in SelectionSpecificity 
        """
        raise NotImplemented("")
    
    def get_weights_per_class(self) -> np.ndarray[np.ndarray]:
        """
            Should be implemented when SelectionMode = WEIGHT and PER_CLASS in SelectionSpecificity 
        """
        raise NotImplemented("")
    
    def get_general_ranking(self) -> np.ndarray:
        """
            Should be implemented when SelectionMode = RANK and GENERAL in SelectionSpecificity 
        """
        raise NotImplemented("")
    
    def get_ranking_per_class(self) -> list[np.ndarray]:
        """
            Should be implemented when SelectionMode = RANK and PER_CLASS in SelectionSpecificity 
        """
        raise NotImplemented("")