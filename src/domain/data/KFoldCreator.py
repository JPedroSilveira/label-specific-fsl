from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from config.type import Config
from src.domain.data.DatasetsCreator import Dataset


class KFoldCreator:
    @staticmethod
    def execute(dataset: Dataset, config: Config) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        if config.dataset.k_fold == 0:
            return [dataset for _ in range(0, config.dataset.k_fold_repeat)]
        datasets = []
        if config.dataset.k_fold_repeat > 1:
            kf = RepeatedStratifiedKFold(n_splits=config.dataset.k_fold, n_repeats=config.dataset.k_fold_repeat, random_state=config.random_seed)
        else:
            kf = StratifiedKFold(n_splits=config.dataset.k_fold, shuffle=True, random_state=config.random_seed)
        X = dataset.get_features()
        y = np.array(dataset.get_labels())
        for train_index, test_index in kf.split(X, y):
            X_train, _ = X[train_index], X[test_index]
            y_train, _ = y[train_index], y[test_index]
            datasets.append(
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