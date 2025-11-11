import numpy as np


from src.domain.selector.types.base.BaseSelector import BaseSelector
from src.domain.selector.types.enum.SelectorSpecificity import SelectorSpecificity

from src.domain.selector.types.enum.SelectorType import SelectorType
from src.evaluation.prediction.prediction_evaluator.BasePredictionEvaluator import BasePredictionEvaluator
from src.domain.data.DatasetsCreator import Dataset


class GeneralPredictionEvaluatorByRank(BasePredictionEvaluator):
    def get_name() -> str:
        return "Rank"
    
    def _get_selection_mode(self) -> SelectorType:
        return SelectorType.RANK
    
    def _get_selecion_specificity(self) -> SelectorSpecificity:
        return SelectorSpecificity.GENERAL
    
    def _get_selected_features(self, selector: BaseSelector, dataset: Dataset, selection_size: int) -> np.ndarray:
        '''
        Given a selection size N, a dataset D and a history item I, returns the N first selected features of D based on the feature selection rank of I
        '''
        feature_rank = selector.get_general_ranking()    
        # Validate if selection size is smaller than the number of available features
        if len(feature_rank) < selection_size:
            raise ValueError("Selection size can't be bigger than the number of available features")
        # Get the selected features indexes
        selected_features = self._get_n_selected_features_from_rank(feature_rank, selection_size)
        # Returns the rows from the dataset with only the selected features
        return self._get_selected_features_from_dataset(dataset, selected_features)
