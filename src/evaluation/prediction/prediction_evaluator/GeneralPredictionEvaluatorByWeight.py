import numpy as np
from src.selector.BaseSelectorWrapper import BaseSelectorWrapper
from src.selector.enum.SelectionMode import SelectionMode
from src.selector.enum.SelectionSpecificity import SelectionSpecificity
from src.evaluation.prediction.prediction_evaluator.BasePredictionEvaluator import BasePredictionEvaluator
from src.history.ExecutionHistory import HistoryItem
from src.domain.data.DatasetsCreator import Dataset


class GeneralPredictionEvaluatorByWeight(BasePredictionEvaluator):
    def get_name() -> str:
        return "Weight"

    def _get_selection_mode(self) -> SelectionMode:
        return SelectionMode.WEIGHT
    
    def _get_selecion_specificity(self) -> SelectionSpecificity:
        return SelectionSpecificity.GENERAL

    def _get_selected_features(self, selector: BaseSelectorWrapper, dataset: Dataset, selection_size: int) -> np.ndarray:
        '''
        Given a selection size N, a dataset D and a history item I, returns the N first selected features of D based on the feature selection weights of I
        '''
        feature_rank = src.selector.get_general_ranking()
        feature_weights = src.selector.get_general_weights()
        # Validate if selection size is smaller than the number of available features
        if len(feature_weights) < selection_size or len(feature_rank) < selection_size:
            raise ValueError("Selection size can't be bigger than the number of available features")
        # Get the selected features indexes based on selection rank (ordered by highest weight)
        selected_features = self._get_n_selected_features_from_rank(feature_rank, selection_size)
        # Get the rows from the dataset with only the selected features
        features = self._get_selected_features_from_dataset(dataset, selected_features)
        # Get the weights from the selected features
        weights = self._get_selected_features_weights(selected_features, feature_weights)
        # Returns the weighed features
        return features * weights