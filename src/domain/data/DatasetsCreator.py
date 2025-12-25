import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config.type import Config
from src.domain.data.types.SplittedDataset import SplittedDataset
from src.util.string_util import extract_numbers


class DatasetsCreator:
    @staticmethod
    def execute(dataframe: pd.DataFrame, config: Config) -> SplittedDataset:
        feature_columns = []
        informative_features = []
        informative_features_per_label = {}
        for index, column in enumerate(dataframe.columns):
            print(column)
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
        features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=test_size, stratify=labels, shuffle=True, random_state=config.dataset.random_seed)
        return SplittedDataset(
            features=features,
            labels=labels,
            features_train=features_train, 
            features_test=features_test,
            labels_train=labels_train,
            labels_test=labels_test,
            label_types=label_types, 
            feature_names=feature_columns,
            informative_features=informative_features, 
            informative_features_per_label=informative_features_per_label
        )