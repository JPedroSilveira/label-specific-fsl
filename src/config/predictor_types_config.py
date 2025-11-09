from typing import Type, List
from src.domain.predictor.types.base.BasePredictor import BasePredictor
from src.domain.predictor.types.SequentialPredictor import SequentialPredictor
from src.domain.predictor.types.SVCPredictor import SVCPredictor
from src.evaluation.prediction.prediction_evaluator.BasePredictionEvaluator import BasePredictionEvaluator
from src.evaluation.prediction.prediction_evaluator.GeneralPredictionEvaluatorByRank import GeneralPredictionEvaluatorByRank
from src.evaluation.prediction.prediction_evaluator.GeneralPredictionEvaluatorByWeight import GeneralPredictionEvaluatorByWeight


PREDICTION_EVALUATOR_TYPES: List[Type[BasePredictionEvaluator]] = [GeneralPredictionEvaluatorByRank, GeneralPredictionEvaluatorByWeight]
PREDICTOR_TYPES: List[Type[BasePredictor]] = [SVCPredictor]#, SequentialPredictor] # [SVCPredictor] #[SequentialPredictor] # SVCPredictor
