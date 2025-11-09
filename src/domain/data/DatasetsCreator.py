import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from config.type import Config
from src.model.Dataset import Dataset
from src.model.SplittedDataset import SplittedDataset
from src.config.general_config import K_FOLD, K_FOLD_REPEAT, RANDOM_STATE
from src.util.string_util import extract_numbers

class DatasetsCreator:
    @staticmethod
    def execute(dataframe: pd.DataFrame, config: Config) -> SplittedDataset:
        feature_columns = []
        informative_features = []
        informative_features_per_label = {}
        for index, column in enumerate(dataframe.columns):
            if column in config.dataset.ignored_columns:
                continue
            if column != config.dataset.label_column:
                feature_columns.append(column)
                if column.startswith(config.feature.informative_prefix):
                    informative_features.append(index)
                if column.startswith(config.feature.informative_per_label_prefix):
                    labels = extract_numbers(column)
                    for label in labels:
                        if label not in informative_features_per_label:
                            informative_features_per_label[label] = [index]
                        else:
                            informative_features_per_label[label].append(index)
        features = dataframe[feature_columns]
        features = features.values
        labels = dataframe[config.dataset.label_column].values
        label_types = np.unique(labels)
        test_size = config.dataset.test_percentage/100
        features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=test_size, stratify=labels, shuffle=True, random_state=config.random_seed)
        return SplittedDataset(
            features_train=features_train, 
            features_test=features_test,
            labels_train=labels_train,
            labels_test=labels_test,
            label_types=label_types, 
            feature_names=feature_columns,
            informative_features=informative_features, 
            informative_features_per_label=informative_features_per_label
        )

def get_train_and_test_data_from_dataset(dataset: Dataset, test_size: int) -> SplittedDataset:
    label_types = dataset.get_label_types()
    feature_names = dataset.get_feature_names()
    informative_features = dataset.get_informative_features()
    informative_features_per_label = dataset.get_informative_features_per_label()
    features = dataset.get_features()
    labels = dataset.get_labels()
    return SplittedDataset(
        train_test_split_output=train_test_split(features, labels, test_size=test_size, stratify=labels, shuffle=True, random_state=RANDOM_STATE),
        features=features,
        labels=labels,
        label_types=label_types,
        feature_names=feature_names,
        informative_features=informative_features,
        informative_features_per_label=informative_features_per_label,
        is_already_pre_processed=True
    )