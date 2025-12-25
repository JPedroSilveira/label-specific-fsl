import statistics
import pandas as pd
from typing import Dict, List, Type

from config.type import Config
from src.domain.data.reader.RankingReader import RankingReader
from src.domain.log.Logger import Logger
from src.domain.storage import ExecutionStorage
from src.domain.data.types.Dataset import Dataset 
from src.domain.selector.types.enum.SelectorSpecificity import SelectorSpecificity
from src.domain.selector.types.base.BaseSelector import BaseSelector


class InformativeFeaturesCalculator:
    @classmethod
    def execute(cls, selectors_class: List[Type[BaseSelector]], dataset: Dataset, config: Config) -> None:
        Logger.execute("Metric: Informative Features Selection (PIFS/PSFI)")
        informative_features = dataset.get_informative_features_names()
        informative_features_by_label = dataset.get_informative_features_names_per_label()
        total_number_of_features = dataset.get_n_features()
        for selector_class in selectors_class:
            ranking_per_specificity = RankingReader.execute(selector_class, config)
            Logger.execute(f"- Selector: {selector_class.get_name()}")
            for specificity in ranking_per_specificity.keys():
                ranking_list = ranking_per_specificity[specificity]
                if len(informative_features) != 0 and specificity == "general":
                    Logger.execute(f"-- Label: {specificity}")
                    cls._calculate_statistics(ranking_list, total_number_of_features, informative_features)
                elif len(informative_features_by_label) != 0 and specificity.startswith("label"):
                    label = int(specificity.replace('label', ''))
                    Logger.execute(f"-- Label: {label}")
                    cls._calculate_statistics(ranking_list, total_number_of_features, informative_features_by_label[label])
                else:
                    Logger.execute(f"-- [INFO] Dataset does not have pre-defined informative features or the specificity ({specificity}) is invalid.")
            
    @classmethod
    def _calculate_statistics(cls, ranking_list: List[pd.DataFrame], total_number_of_features: int, informative_features: List[int]) -> None:
        for i in range(0, total_number_of_features):
            number_of_features = i + 1
            Logger.execute(f"--- Number of features: {number_of_features}")
            pifs_list = []
            psfi_list = []
            for ranking in ranking_list:
                pifs = cls._calculate_pifs(number_of_features, ranking['feature'].to_numpy(), informative_features)
                pifs_list.append(pifs)
                psfi = cls._calculate_psfi(number_of_features, ranking['feature'].to_numpy(), informative_features)
                psfi_list.append(psfi)
            Logger.execute(f"---- PIFS: {pifs_list}")
            Logger.execute(f"----- Mean: {statistics.mean(pifs_list)}")
            Logger.execute(f"----- Standard deviation: {statistics.stdev(pifs_list)}")
            Logger.execute(f"---- PSFI: {psfi_list}")
            Logger.execute(f"----- Mean: {statistics.mean(psfi_list)}")
            Logger.execute(f"----- Standard deviation: {statistics.stdev(psfi_list)}")
                
    def _calculate_pifs(number_of_features: int, ranking: list[int], informative_features: list[int]) -> float:
        selected_features = ranking[0:number_of_features]
        n_informative_selected = len(set(selected_features).intersection(set(informative_features)))
        pifs = n_informative_selected / len(informative_features)
        return pifs
    
    def _calculate_psfi(number_of_features: int, ranking: list[int], informative_features: list[int]) -> float:
        selected_features = ranking[0:number_of_features]
        n_informative_selected = len(set(selected_features).intersection(set(informative_features)))
        psfi = n_informative_selected / len(selected_features)
        return psfi