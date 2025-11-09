import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from src.config.general_config import K_FOLD, K_FOLD_REPEAT, RANDOM_STATE, INFORMATIVE_FEATURE_PREFIX, INFORMATIVE_FEATURE_PER_LABEL_PREFIX, SHOULD_SCALE, SHOULD_STANDARDIZE
from src.data.Dataset import Dataset
from src.util.string_util import extract_numbers


class SplittedDataset():
    def __init__(self, train_test_split_output: list, features: np.ndarray, labels: np.ndarray, label_types: list, feature_names: list[str], informative_features: list[int], informative_features_per_label: dict[int, list[int]], is_already_pre_processed: bool=False) -> None:
        all_features = features
        all_labels = labels
        features_train = train_test_split_output[0]
        features_test = train_test_split_output[1]
        labels_train = train_test_split_output[2]
        labels_test = train_test_split_output[3]
        if not is_already_pre_processed:
            if SHOULD_SCALE:
                scaler = preprocessing.MinMaxScaler()
                features_train = scaler.fit_transform(features_train)
                features_test = scaler.fit_transform(features_test)
            if SHOULD_STANDARDIZE:
                scaler = preprocessing.StandardScaler()
                features_train = scaler.fit_transform(features_train)
                features_test = scaler.fit_transform(features_test)
        self._all_dataset = Dataset(
            features=all_features,
            labels=all_labels,
            label_types=label_types, 
            feature_names=feature_names,
            informative_features=informative_features, 
            informative_features_per_label=informative_features_per_label
        )
        self._train_dataset = Dataset(
            features=features_train, 
            labels=labels_train,
            label_types=label_types, 
            feature_names=feature_names,
            informative_features=informative_features, 
            informative_features_per_label=informative_features_per_label
        )
        self._test_dataset = Dataset(
            features=features_test,
            labels=labels_test,
            label_types=label_types, 
            feature_names=feature_names,
            informative_features=informative_features, 
            informative_features_per_label=informative_features_per_label
        )
        self._labels_types = label_types
        self._informative_features = informative_features
        self._feature_names = feature_names
        self._informative_features_per_label = informative_features_per_label
        if self._train_dataset.get_n_features() != self._test_dataset.get_n_features():
            raise ValueError("Train and test datasets should have the same number of features")
        if self._train_dataset.get_n_labels() != self._test_dataset.get_n_labels():
            raise ValueError("Train and test datasets should have the same number of labels")
        
    def get_all(self):
        return self._all_dataset

    def get_train(self):
        return self._train_dataset
    
    def get_test(self):
        return self._test_dataset
    
    def get_n_features(self):
        return self._train_dataset.get_n_features()
    
    def get_n_labels(self):
        return self._test_dataset.get_n_labels()
    
    def get_label_types(self) -> list:
        return self._labels_types
        
    def get_feature_names(self):
        return self._feature_names
    
    def get_informative_features(self):
        return self._informative_features
    
    def get_informative_features_per_label(self):
        return self._informative_features_per_label

def get_train_and_test_data_from_dataframe(dataframe: pd.DataFrame, test_size: int) -> SplittedDataset:
    ignore_columns = ['samples']
    label_column = ['class']
    feature_columns = []
    informative_features = []
    informative_features_per_label = {}
    for index, column in enumerate(dataframe.columns):
        if column in ignore_columns:
            continue
        if column not in label_column:
            feature_columns.append(column)
            if column.startswith(INFORMATIVE_FEATURE_PREFIX):
                informative_features.append(index)
            if column.startswith(INFORMATIVE_FEATURE_PER_LABEL_PREFIX):
                labels = extract_numbers(column)
                for label in labels:
                    if label not in informative_features_per_label:
                        informative_features_per_label[label] = [index]
                    else:
                        informative_features_per_label[label].append(index)
    features = dataframe[feature_columns]
    features = features.values
    labels = dataframe[label_column]
    labels = [value[0] for value in labels.values]
    label_types = np.unique(labels)
    return SplittedDataset(
        train_test_split_output=train_test_split(features, labels, test_size=test_size, stratify=labels, shuffle=True, random_state=RANDOM_STATE), 
        features=features,
        labels=labels,
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

def get_dataset_with_k_fold(dataset: Dataset, n_splits: int, n_repeats: int) -> list[Dataset]:
    if K_FOLD == 0:
        return [dataset for _ in range(0, K_FOLD_REPEAT)]
    datasets = []
    if K_FOLD_REPEAT > 1:
        kf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE)
    else:
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    X = dataset.get_features()
    y = np.array(dataset.get_labels())
    for train_index, test_index in kf.split(X, y):
        X_train, _ = X[train_index], X[test_index]
        y_train, _ = y[train_index], y[test_index]
        src.datasets.append(
            Dataset(
                features=X_train,
                labels=y_train,
                label_types=dataset.get_label_types(), 
                feature_names=dataset.get_feature_names(),
                informative_features=dataset.get_informative_features(), 
                informative_features_per_label=dataset.get_informative_features_per_label()
            )
        )
    return datasets

