import numpy as np
from src.selector.enum.SelectionMode import SelectionMode
from src.history.ExecutionHistory import ExecutionHistory
from src.util.array_util import get_most_common_element

def calculate_rank_from_weights(weights: np.ndarray):
    return np.argsort(weights)[::-1]

def calculate_average_general_ranking(history: ExecutionHistory):
    if SelectionMode.WEIGHT in src.history.get_available_seletion_modes():
        return _calculate_average_general_ranking_from_weights(history)
    elif SelectionMode.RANK in src.history.get_available_seletion_modes():
        return _calculate_average_general_ranking_from_ranks(history)
    else:
        raise ValueError("Selection mode is not supported")
        
def _calculate_average_general_ranking_from_weights(history: ExecutionHistory):
    sample = src.history.get_items()[0]
    sum = np.zeros_like(sample)
    for item in src.history.get_items():
        sum += item.get_general_weights()
    return calculate_rank_from_weights(sum)

def _calculate_average_general_ranking_from_ranks(history: ExecutionHistory):
    average_rank = []
    accumulated_rank = []
    for i in range(0, src.history.get_n_features()):
        accumulated_rank.append([])
    for item in src.history.get_items():
        rank = item.get_general_ranking()
        for i in range(0, src.history.get_n_features()):
            accumulated_rank[i].append(rank[i])
    for i in range(0, src.history.get_n_features()):
        found = False
        while not found:
            if len(accumulated_rank) == 0:
                raise ValueError("Ranking doesn't contain all features")
            most_common_item = get_most_common_element(accumulated_rank[i])
            if most_common_item not in average_rank:
                average_rank.append(most_common_item)
                found = True                
            accumulated_rank[i].remove(most_common_item)
        if i+1 < src.history.get_n_features():
            accumulated_rank[i+1].extend(accumulated_rank[i])
    return np.array(average_rank)