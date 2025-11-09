import os
import numpy as np

from itertools import chain
from dataclasses import dataclass, field
from sklearn.datasets import make_classification
from sklearn.preprocessing import minmax_scale
from custom_make_classification import custom_make_classification
from multi_class_generator import generate_multi_class_synthetic_dataset

DATASETS_PATH = './datasets/synthetic'
random_state = 44

@dataclass
class DatasetParams:
    n_samples: int
    n_features: int
    n_informative: int = 0
    n_redundant: int = 0
    n_repeated: int = 0
    n_noisy: int = field(init=False)
    n_informative_per_class: int = 0
    n_classes: int = 2

    def __post_init__(self):
        self.n_noisy = self.n_features - self.n_informative - self.n_redundant - self.n_repeated

    def is_multi_class(self):
        return self.n_classes > 2

    def name(self):
        _name = "synth_"
        _name += f"{self.n_classes}classes_"
        _name += f"{self.n_samples}samples_{self.n_features}features"
        if self.n_informative > 0:
            _name += f"_{self.n_informative}informative"
        if self.is_multi_class() and self.n_informative_per_class > 0:
            _name += f"_{self.n_informative_per_class}informativeperclass"
        if self.n_redundant > 0:
            _name += f"_{self.n_redundant}redundant"
        if self.n_repeated > 0:
            _name += f"_{self.n_repeated}repeated"
        return _name

    def csv_name(self):
        return self.name() + '.csv'

    def feature_names(self):
        types = chain(
            ("informative" for _ in range(self.n_informative)),
            ("redundant" for _ in range(self.n_redundant)),
            ("repeated" for _ in range(self.n_repeated)),
            ("noisy" for _ in range(self.n_noisy))
        )
        return [f'{t}_{x}' for t, x in zip(types, range(self.n_features))]
    
    def feature_names_multi_class(self, features_by_class):
        feature_names = []
        for i in range(0, self.n_features):
            classes = []
            is_relevant = False
            for c in range(0, self.n_classes):
                if i in features_by_class[c]:
                    is_relevant = True
                    classes.append("_" + str(c))

            if is_relevant:
                feature_names.append("informative_for_labels_" + self.int_to_alpha(i) + "".join(classes))
            else:
                feature_names.append("noisy" + self.int_to_alpha(i))
        return feature_names

    def int_to_alpha(self, num):
        num += 1
        if num <= 0:
            raise ValueError("Number must be positive")

        result = ""
        while num > 0:
            remainder = (num - 1) % 26
            result = chr(ord('A') + remainder) + result
            num = (num - 1) // 26

        return result

    def build_dataset(self):
        if self.is_multi_class() and self.n_informative_per_class > 0:
            return self.build_multi_class_dataset()
        else:
            return self.build_dataset_using_all_informative_for_all_classes()
    
    def build_multi_class_dataset(self):
        X, y, features_by_class = custom_make_classification(
            n_samples=self.n_samples, 
            n_features=self.n_features, 
            n_informative=self.n_informative,
            n_informative_features_per_class=self.n_informative_per_class,
            n_classes=self.n_classes,
            random_state=random_state,
            shuffle=False)
        
        return X, y, self.feature_names_multi_class(features_by_class)

    def build_dataset_using_all_informative_for_all_classes(self):
        X, y = make_classification(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_informative=self.n_informative,
            n_redundant=self.n_redundant,
            n_repeated=self.n_repeated,
            n_classes=self.n_classes,
            shuffle=False,
            random_state=random_state
        )
        return X, y, self.feature_names()

    def to_csv(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        X, y, cols = self.build_dataset()
        data = np.column_stack((X, y))
        sepcs = ['%.10f' for _ in cols] + ['%i']
        cols += ['class']
        header = ','.join(f'"{c}"' for c in cols)
        path_to_save = os.path.join(path, self.csv_name())
        np.savetxt(
            path_to_save,
            data,
            fmt=sepcs,
            header=header,
            delimiter=',',
            comments=''
        )
        print(f"Saved dataset to {path_to_save}")


datasets = [
    DatasetParams(n_classes=3, n_samples=3000, n_features=100, n_informative=30),
    DatasetParams(n_classes=3, n_samples=3000, n_features=300, n_informative=30, n_informative_per_class=10),
    DatasetParams(n_classes=3, n_samples=3000, n_features=300, n_informative=30, n_informative_per_class=15),
]

for dataset in datasets:
    dataset.to_csv(DATASETS_PATH)