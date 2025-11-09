import numpy as np
from sklearn.model_selection import StratifiedKFold
import src.config.general_config as general_config
from typing import Type
from src.util.classification_report_util import calculate_classification_report
from src.history.ExecutionHistory import HistoryItem
from src.data.DataSplitter import SplittedDataset, Dataset
from src.selector.BaseSelectorWrapper import BaseSelectorWrapper
from src.selector.enum.SelectionMode import SelectionMode, is_compatible_mode
from src.selector.enum.SelectionSpecificity import SelectionSpecificity
from src.predictor.BasePredictor import BasePredictor
from src.evaluation.prediction.PredictionScore import PredictorPredictionScore
from src.util.feature_selection_util import get_n_features_from_rank
from src.util.performance_util import ExecutionTimeCounter
from src.util.print_util import print_with_time


class BasePredictionEvaluator():
    '''
    Define all based methods that an evaluator should implement
    '''
    def __init__(self, selector: BaseSelectorWrapper, selection_size: int, predictor_type: Type[BasePredictor]):
        self._selector = selector
        self._selection_size = selection_size
        self._predictor_type = predictor_type

    def get_name() -> str:
        """
        Return a identifier for the evaluation type
        """
        raise NotImplemented("")
    
    def get_class_name(self) -> str:
        return self.__class__.get_name()
    
    def _get_selection_mode(self) -> SelectionMode:
        """
        Return the selection mode which will be evaluated
        """
        raise NotImplemented("")
    
    def _get_selecion_specificity(self) -> SelectionSpecificity:
        """
        Return the selection specificity which will be evaluated
        """
        raise NotImplemented("")

    def _get_selected_features(self, selector: BaseSelectorWrapper, dataset: Dataset, selection_size: int) -> np.ndarray:
        raise NotImplemented("")
    
    def should_execute(self, selector: BaseSelectorWrapper):
        '''
        Verify if an evaluation should be execute based on selector capabilities.
        Example.: Given a selector that only creates a general rank, it can not be evaluated by an evaluator that considers per label specific ranks
        '''
        return is_compatible_mode(self._get_selection_mode(), src.selector.get_selection_mode()) and self._get_selecion_specificity() in src.selector.get_selection_specificities()

    def calculate(self, selector: BaseSelectorWrapper, splitted_dataset: SplittedDataset) -> list[PredictorPredictionScore]:
        '''
        Given a selector execution (HistoryItem), calculates a predictor average score based on the feature selector 
        '''
        #print_with_time(f'{self.get_class_name()} - Calculating predictor performance')
        #evaluation_timer = ExecutionTimeCounter().start()
        scores: list[PredictorPredictionScore] = []
        # Calculates the predictor scores training it multiple times
        for _ in range(0, general_config.PREDICTOR_EXECUTIONS):
            #round_timer = ExecutionTimeCounter().start()
            #print_with_time(f'Round {round}')
            # Define train data based on N selected features
            x_train = self._get_selected_features(selector, splitted_dataset.get_train(), self._selection_size)
            predictor = self._predictor_type(self._selection_size, splitted_dataset.get_n_labels())
            y_train = self._get_labels_for_predictor(splitted_dataset.get_train(), predictor)
            # Train model
            src.predictor.fit(x_train, y_train)
            # Define the test features based on N selected features
            x_test = self._get_selected_features(selector, splitted_dataset.get_test(), self._selection_size)
            # Test model
            y_pred = src.predictor.predict(x_test)
            # Calculate performance metrics based on predictions
            report = calculate_classification_report(splitted_dataset.get_test(), y_pred)
            # Save score on list
            score = PredictorPredictionScore(self._selection_size, self._selector, predictor, self.get_class_name(), report)            
            scores.append(score)
            #round_timer.print('Predictor training and test round')
        #evaluation_timer.print('Predictor evalution')
        # Return list with all scores based on multiple executions
        return scores

    def _get_labels_for_predictor(self, dataset: Dataset, predictor: BasePredictor):
        '''
        Returns the labels based on predictor preference (raw or encoded with OneHotEncoder)
        '''
        return dataset.get_encoded_labels() if src.predictor.should_encode_labels() else dataset.get_labels()

    def _get_n_selected_features_from_rank(self, feature_rank: np.ndarray, selection_size: int) -> list[int]:
        '''
        Get the first n features from the selecion rank
        '''
        return get_n_features_from_rank(feature_rank, selection_size)
    
    def _get_n_selected_features_from_weights(self, feature_weights: np.ndarray, selection_size: int) -> list[int]:
        '''
        Get the first n features from teh selection weights ordered by highest weight value
        '''
        selected_features = np.argsort(feature_weights)[::-1][:selection_size]
        return selected_features
    
    def _get_selected_features_from_dataset(self, dataset: Dataset, selected_features: list[int]):
        '''
        Given a list of selected features indexes and a dataset, returns the feature raws with only the selected features
        '''
        features = dataset.get_features()
        return features[:, selected_features]
    
    def _get_selected_features_weights(self, selected_features: list[int], feature_weights: np.ndarray):
        '''
        Given a list of selected features indexes and a list with all feature weights, returns the weight of all selected features
        '''
        return feature_weights[selected_features]