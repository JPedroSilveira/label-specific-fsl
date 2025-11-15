import random
import numpy as np
from sklearn.utils import shuffle as util_shuffle
from sklearn.utils.validation import check_random_state

def generate_multi_class_synthetic_dataset(
    n_features: int,
    n_informative: int, 
    n_informative_features_per_class: int,
    n_samples: int,
    n_classes: int,
    random_state: int,
    shuffle: bool
):
    n_useless = n_features - n_informative
    generator = check_random_state(random_state)
    features_by_class = []
    available_informative_features = list(range(0, n_informative))
    available_informative_features = util_shuffle(available_informative_features, random_state=generator)
    divided_features, remainder_list = _divide_list(available_informative_features, n_classes)
    for features in divided_features:
        features_by_class.append(features)

    index_by_class = []
    for label in range(0, n_classes):
        index_by_class.append(0)
    current_copy_class = 0
    for label in range(0, n_classes):
        while len(features_by_class[label]) < n_informative_features_per_class:
            if current_copy_class == label:
                current_copy_class += 1
            if current_copy_class >= n_classes:
                current_copy_class = 0
            index_to_copy = index_by_class[current_copy_class]
            index_by_class[current_copy_class] += 1
            value_to_copy = features_by_class[current_copy_class][index_to_copy]
            features_by_class[label].append(value_to_copy)
            current_copy_class += 1
            if current_copy_class >= n_classes:
                current_copy_class = 0

    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples)

    samples = util_shuffle(list(range(0, len(X))))
    samples_by_class, remainder_list = _divide_list(samples, n_classes)
    for current_class, samples in enumerate(samples_by_class):
        class_features = features_by_class[current_class]
        non_class_features = [i for i in range(0, n_informative) if i not in class_features]
        y[samples] = current_class
        for sample in samples:
            X[sample, class_features] = 1
            X[sample, non_class_features] = _generate_random_binary_list(len(non_class_features))

    current_class = 0
    for sample in remainder_list:
        class_features = features_by_class[current_class]
        non_class_features = [i for i in range(0, n_informative) if i not in class_features]
        y[sample] = current_class
        X[sample, class_features] = 1
        X[sample, non_class_features] = _generate_random_binary_list(len(non_class_features))
        current_class += 1
        if current_class >= n_classes:
            current_class = 0

    if n_useless > 0:
        X[:, -n_useless:] = _generate_random_binary_matrix(n_samples, n_useless)

    if shuffle:
        X, y = util_shuffle(X, y, random_state=generator)

    for sample in range(0, n_samples):
        x = X[sample]
        current_class = y[sample]
        informative_features = features_by_class[int(current_class)]
        for feature in informative_features:
            if x[feature] != 1:
                raise ValueError(f"Sample has invalid combination of features based on class")
            
    return X, y, features_by_class

def _generate_random_binary_list(length):
  return [random.randint(0, 1) for _ in range(length)]

def _generate_random_binary_matrix(rows, cols):
  return np.random.randint(2, size=(rows, cols))

def _divide_list(list, num_divisions):
  num_per_list = len(list) // num_divisions
  remainder = len(list) % num_divisions

  divided_lists = []
  for i in range(num_divisions):
    start_idx = i * num_per_list
    end_idx = start_idx + num_per_list
    divided_lists.append(list[start_idx:end_idx])

  remainder_list = []
  if remainder > 0:
    remainder_list = list[-remainder:]

  return divided_lists, remainder_list