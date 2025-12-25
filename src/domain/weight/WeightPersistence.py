from typing import List
import numpy as np
import pandas as pd

from config.type import OutputConfig
from src.domain.data.Normalizer import Normalizer
from src.domain.selector.types.enum.SelectorType import SelectorType
from src.domain.selector.types.base.BaseSelector import BaseSelector, SelectorSpecificity


class WeightPersistence:
    @classmethod
    def execute(cls, id: int, selector: BaseSelector, config: OutputConfig, feature_names: List[str]) -> None:
        if selector.get_type() == SelectorType.WEIGHT:
            cls._persist_general_weights(id, selector, config, feature_names)
            if selector.get_specificity() == SelectorSpecificity.PER_LABEL:
                cls._persist_per_label_weights(id, selector, config, feature_names)
                
    @classmethod
    def _persist_general_weights(cls, id: int, selector: BaseSelector, config: OutputConfig, feature_names: List[str]) -> None:
        weights = selector.get_general_weights()
        filename = f"{config.execution_output.raw_selection}/{str(id)}_{selector.get_selector_name().lower()}_general_weights"
        cls._persist_weights(weights, filename, feature_names)
    
    @classmethod
    def _persist_per_label_weights(cls, id: int, selector: BaseSelector, config: OutputConfig, feature_names: List[str]) -> None:
        weights_per_label = selector.get_per_label_weights()
        for label, weights in enumerate(weights_per_label):
            filename = f"{config.execution_output.raw_selection}/{str(id)}_{selector.get_selector_name().lower()}_label{str(label)}_weights"
            cls._persist_weights(weights, filename, feature_names)
    
    @classmethod
    def _persist_weights(cls, weights: np.ndarray, filename: str, feature_names: List[str]) -> None:
        df_raw = cls._persist_weights_raw(weights, filename, feature_names)
        cls._persist_weights_normalized(df_raw, filename)
        cls._persist_weights_ranked(df_raw, filename)
        
    @staticmethod
    def _persist_weights_raw(weights: np.ndarray, filename: str, feature_names: List[str]) -> pd.DataFrame:
        df_raw = pd.DataFrame(
            list(zip(feature_names, weights)), 
            columns=["feature", "value"]
        )
        df_raw.to_csv(f"{filename}_raw.csv", index=False)
        return df_raw
    
    @staticmethod
    def _persist_weights_normalized(df_raw: pd.DataFrame, filename: str) -> None:
        df_normalized = df_raw.copy()
        df_normalized['normalized_value'] = Normalizer.execute(df_normalized['value'])
        df_normalized = df_normalized.drop(columns=['value']).rename(
            columns={'normalized_value': 'value'}
        )
        df_normalized.to_csv(f"{filename}_normalized.csv", index=False)
        
    @staticmethod
    def _persist_weights_ranked(df_raw: pd.DataFrame, filename: str) -> None:
        df_sorted = df_raw.copy().sort_values(by='value', ascending=False).reset_index(drop=True)
        df_sorted.to_csv(f"{filename}_sorted.csv", index=False)