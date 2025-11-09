import numpy as np


def get_weight_per_class_from_shap(shap_values):
    """
    Given a array of shap values in the following format where F represents the number of features and C the number of classes

    [
        [0 ... C] 0 
        [0 ... C] 1
        ...
        [0 ... C] F
    ]

    Returns a new array on the format:

    [
        [0 ... F] 0
        ...
        [0 ... F] C
    ]
    """
    print(shap_values.shape)
    n_labels = len(shap_values[0])
    transpose = []
    for i in range(0, n_labels):
        transpose.append([])
    for feature_per_class in shap_values:
        for i in range(0, n_labels):
            transpose[i].append(feature_per_class[i])
    for i in range(0, n_labels):
        transpose[i] = np.array(transpose[i], dtype=np.float64)
    return transpose


def get_most_common_element(lst):
    if not lst:
        raise ValueError("List cannot be empty.")

    counts = {}
    for element in lst:
        counts[element] = counts.get(element, 0) + 1

    most_common = max(counts, key=counts.get)
    return most_common

def divide_chunks(lst, n):
    if n <= 0:
        raise ValueError("n must be positive")

    if n > len(lst):
        return [lst]

    sublist_size = len(lst) // n
    sublists = []

    for i in range(n):
        start_idx = i * sublist_size
        end_idx = start_idx + sublist_size 
        sublists.append(lst[start_idx:end_idx])
    count = 0
    for i in range(end_idx, len(lst)):
        sublists[count].append(lst[i])
        count +=1

    return sublists

def filter_list(list1, list2):
    filtered_list = []
    for l1 in list1:
        if l1 in list2:
            filtered_list.append(l1)
    return filtered_list