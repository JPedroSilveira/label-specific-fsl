from typing import Type
from src.predictor.BasePredictor import BasePredictor
from src.predictor.SequentialPredictor import SequentialPredictor
from src.predictor.SVCPredictor import SVCPredictor
from typing import Type
from src.evaluation.prediction.prediction_evaluator.BasePredictionEvaluator import BasePredictionEvaluator
from src.evaluation.prediction.prediction_evaluator.GeneralPredictionEvaluatorByRank import GeneralPredictionEvaluatorByRank
from src.evaluation.prediction.prediction_evaluator.GeneralPredictionEvaluatorByWeight import GeneralPredictionEvaluatorByWeight


PREDICTION_EVALUATOR_TYPES: list[Type[BasePredictionEvaluator]] = [GeneralPredictionEvaluatorByRank, GeneralPredictionEvaluatorByWeight]
PREDICTOR_TYPES: list[Type[BasePredictor]] = [SVCPredictor]#, SequentialPredictor] # [SVCPredictor] #[SequentialPredictor] # SVCPredictor
