import numpy as np
from typing import Dict, List, Self

from src.domain.data.DatasetNormalizer import Dataset


class FeatureRemover:
    @classmethod
    def execute(cls, dataset: Dataset, feature_name: str) -> Dataset:
        current_names = dataset.get_feature_names()
        index_to_remove = cls._get_feature_index(current_names, feature_name)
        new_features = cls._get_new_features(dataset, index_to_remove)
        new_feature_names = cls._get_new_feature_names(current_names, index_to_remove)
        new_informative_features = cls._get_new_informative_features(dataset, index_to_remove)
        new_informative_features_per_label = cls._get_new_informative_features_per_label(dataset, index_to_remove)
        return Dataset(
            features=new_features,
            labels=dataset.get_labels(),
            label_types=dataset.get_label_types(),
            feature_names=new_feature_names,
            informative_features=new_informative_features,
            informative_features_per_label=new_informative_features_per_label
        )
        
        
    @staticmethod
    def _get_feature_index(current_names: List[str], feature_name: str) -> int:
        try:
            return current_names.index(feature_name)
        except ValueError:
            raise ValueError(f"Feature name '{feature_name}' not found in the dataset.")
        
    @staticmethod
    def _get_new_feature_names(current_names: List[str], index_to_remove: int) -> List[str]:
        return current_names[:index_to_remove] + current_names[index_to_remove+1:]
    
    @staticmethod
    def _get_new_features(dataset: Dataset, index_to_remove: int) -> np.ndarray:
        return np.delete(dataset.get_features(), index_to_remove, axis=1)
    
    @classmethod
    def _get_new_informative_features(cls, dataset: Dataset, index_to_remove: int) -> List[int]:
        return cls._remove_informative_feature(dataset.get_informative_features(), index_to_remove)
    
    @classmethod
    def _get_new_informative_features_per_label(cls, dataset: Dataset, index_to_remove: int) -> Dict[int, List[int]]:
        new_informative_features_per_label = {}
        if dataset.get_informative_features_per_label() is None:
            return None
        for label, old_indexes in dataset.get_informative_features_per_label().items():
            new_informative_features_per_label[label] = cls._remove_informative_feature(old_indexes, index_to_remove)
        return new_informative_features_per_label
        
            
    @staticmethod
    def _remove_informative_feature(old_indexes: List[int], index_to_remove: int) -> List[int]:
        new_indexes: List[int] = []
        for old_index in old_indexes:
            if old_index == index_to_remove:
                continue
            elif old_index > index_to_remove:
                new_index = old_index - 1
            else:
                new_index = old_index
            new_indexes.append(new_index)
        return new_indexes