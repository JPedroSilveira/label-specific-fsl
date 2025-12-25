import os
import pandas as pd
from typing import Dict, List, Type
from pandas import DataFrame

from config.type import Config
from src.domain.selector.types.base.BaseSelector import BaseSelector


class Reader:
    @classmethod
    def execute(cls, selector_class: Type[BaseSelector], config: Config) -> Dict[str,List[DataFrame]]:
        valuePerSpecificity: Dict[str,List[DataFrame]] = {}
        all_selection_files: List[str] = os.listdir(config.output.execution_output.raw_selection)
        all_selection_files.sort()
        for selection_file in all_selection_files:
            if selection_file.endswith(cls._get_file_suffix()):
                filename_parts = selection_file.split('_')
                selector_type = filename_parts[1]
                specificit = filename_parts[2]
                if selector_type == selector_class.get_name().lower():
                    new_weights = cls._read_values_from_file(selection_file, config)
                    if specificit in valuePerSpecificity:
                        valuePerSpecificity[specificit].append(new_weights)
                    else:
                        valuePerSpecificity[specificit] = [new_weights]
        return valuePerSpecificity
        
    @staticmethod
    def _read_values_from_file(filename: str, config: Config) -> DataFrame:
        return pd.read_csv(f"{config.output.execution_output.raw_selection}/{filename}")
    
    @staticmethod
    def _get_file_suffix() -> str:
        raise NotImplementedError("Subclasses must implement this method")