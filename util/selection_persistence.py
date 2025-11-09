import csv
import numpy as np
from typing import List
from evaluation.prediction.PredictionScore import SelectorPredictionScore
from config.general_config import OUTPUT_PATH, RAW_SELECTION_SUBPATH
from data.Dataset import Dataset
from history.ExecutionHistory import ExecutionHistory
from selector.enum.SelectionMode import SelectionMode
from selector.enum.SelectionSpecificity import SelectionSpecificity
from util.numpy_util import normalize, sort
from util.print_util import print_with_time
from util.performance_util import ExecutionTimeCounter
from util.dict_util import add_on_dict_list


def _write_to_csv_file(data: list[list], filename: str):
    '''
    Write data to CSV
    '''
    with open(f'{OUTPUT_PATH}/{RAW_SELECTION_SUBPATH}/{filename}.csv', "w", newline="") as csvfile:
        # Start CSV writer
        csvwriter = csv.writer(csvfile)
        # Write the header row
        csvwriter.writerow(data[0])  
        # Write the data rows
        csvwriter.writerows(data[1:])

def _write_dict_to_csv_file(dict_data: dict[str, List], header: List[str], filename: str):
    '''
    Write data to CSV
    '''
    data = []
    for key in dict_data.keys():
        data.append([key, dict_data.get(key)])
    with open(f'{OUTPUT_PATH}/{RAW_SELECTION_SUBPATH}/{filename}.csv', "w", newline="") as csvfile:
        # Start CSV writer
        csvwriter = csv.writer(csvfile)
        # Write the header row
        csvwriter.writerow(header)  
        # Write the data rows
        csvwriter.writerows(data)

def _persist_selection(filename: str, selections: list, feature_names: List[str], is_weight=False):
    '''
    Persist weights in TSNE format
    '''
    data = []
    for index, selection in enumerate(selections):
        if is_weight:
            data.append([feature_names[index], selection])
        else:
            data.append([feature_names[selection], index])
    # Sort features by weight
    #data = sorted(data, key=lambda x: x[1], reverse=is_weight)
    # Add header
    data.insert(0, ["feature", "value"])
    # Write values to csv
    _write_to_csv_file(data, filename)

def _persist_per_label_weights(history: ExecutionHistory, feature_names: list[str], best_score_index: int):
    '''
    Persist weights per label in TSNE format
    '''
    # Persist a file for each execution
    for index, item in enumerate(history.get_items()):
        ranks = item.get_rank_per_class()
        weights = item.get_weights_per_class()
        # Persist a file for each label
        for label, label_weights in enumerate(weights):
            rank = ranks[label]
            isBestScore = best_score_index == index
            filename = f"{history.get_selector_name().lower()}-weights-by-label-{str(label)}-item-{str(index)}"
            if isBestScore:
                filename = f"{filename}-best-score"
            # Persist weights
            _persist_selection(
                filename=filename,
                selections=label_weights,
                feature_names=feature_names,
                is_weight=True
            )
            # Ordered weight
            sorted_weights = [label_weights[i] for i in rank]
            sorted_feature_names = [feature_names[i] for i in rank]
            _persist_selection(
                filename=f"{filename}-sorted",
                selections=sorted_weights,
                feature_names=sorted_feature_names,
                is_weight=True
            )
            # Persist normalized weights
            _persist_selection(
                filename=f"{filename}-normalized",
                selections=normalize(label_weights),
                feature_names=feature_names,
                is_weight=True
            )
            

def _persist_general_weights(history: ExecutionHistory, feature_names: list[str], best_score_index: int):
    '''
    Persist general weights in TSNE format
    '''
    # Persist a file for each execution
    for index, item in enumerate(history.get_items()):
        isBestScore = best_score_index == index
        filename = f"{history.get_selector_name().lower()}-weights-general-item-{str(index)}"
        if isBestScore:
            filename = f"{filename}-best-score"
        weights = item.get_general_weights()
        rank = item.get_general_ranking().tolist()
        _persist_selection(
            filename=filename,
            selections=weights,
            feature_names=feature_names,
            is_weight=True
        )
        # Ordered weight
        sorted_weights = [weights[i] for i in rank]
        sorted_feature_names = [feature_names[i] for i in rank]
        _persist_selection(
            filename=f"{filename}-sorted",
            selections=sorted_weights,
            feature_names=sorted_feature_names,
            is_weight=True
        )
        # Persist normalized weights
        filename = f"{filename}-normalized"
        _persist_selection(
            filename=filename,
            selections=normalize(weights),
            feature_names=feature_names,
            is_weight=True
        )

def _persist_per_label_ranks(history: ExecutionHistory, feature_names: list[str], best_score_index: int):
    '''
    Persist ranks per label
    '''
    # Persist a file for each execution
    for index, item in enumerate(history.get_items()):
        isBestScore = best_score_index == index
        ranks = item.get_rank_per_class()
        # Persist a file for each label
        for label, rank in enumerate(ranks):
            filename = f"{history.get_selector_name().lower()}-rank-by-label-{str(label)}-item-{str(index)}"
            if isBestScore:
                filename = f"{filename}-best-score"
            # Persist weights
            _persist_selection(
                filename=filename,
                selections=rank,
                feature_names=feature_names,
                is_weight=False
            )


def _persist_general_ranks(history: ExecutionHistory, feature_names: list[str], best_score_index: int):
    '''
    Persist general weights in TSNE format
    '''
    # Persist a file for each execution
    for index, item in enumerate(history.get_items()):
        isBestScore = best_score_index == index
        filename = f"{history.get_selector_name().lower()}-rank-general-item-{str(index)}"
        if isBestScore:
            filename = f"{filename}-best-score"
        rank = item.get_general_ranking()
        _persist_selection(
            filename=filename,
            selections=rank,
            feature_names=feature_names,
            is_weight=False
        )

def persist_rank(history: ExecutionHistory, feature_names: list[str], best_score_index: int):
    '''
    Persist all executions rank
    '''
    # Verify if ranks are available
    if SelectionMode.RANK not in history.get_available_seletion_modes():
        return
    print_with_time("Persisting ranks...")
    time_count = ExecutionTimeCounter().start()
    if SelectionSpecificity.GENERAL in history.get_selection_specificities():
        _persist_general_ranks(history, feature_names, best_score_index)
    if SelectionSpecificity.PER_LABEL in history.get_selection_specificities():
        _persist_per_label_ranks(history, feature_names, best_score_index)
    time_count.print_end("Rank persistence")

def persist_weights(history: ExecutionHistory, feature_names: list[str], best_score_index: int):
    '''
    Persist all executions weights in TSNE format
    '''
    # Verify if weights are available
    if SelectionMode.WEIGHT not in history.get_available_seletion_modes():
        return
    print_with_time("Persisting weights...")
    time_count = ExecutionTimeCounter().start()
    if SelectionSpecificity.GENERAL in history.get_selection_specificities():
        _persist_general_weights(history, feature_names, best_score_index)
    if SelectionSpecificity.PER_LABEL in history.get_selection_specificities():
        _persist_per_label_weights(history, feature_names, best_score_index)
    time_count.print_end("Weight persistence")

def persist_execution_metrics(execution_times_by_selector: dict[str, List[float]], 
                              selector_prediction_scores_by_selector: dict[str, List[SelectorPredictionScore]] = {}):
    # Execution time
    data = []
    for selector in execution_times_by_selector.keys():
        execution_times = execution_times_by_selector.get(selector)
        data.append([selector, execution_times])
    data.insert(0, ["selector", "time"])
    _write_to_csv_file(data, f'raw-execution-times')
    # Predictor scores
    f1_data = []
    accuracy_data = []
    precision_data = []
    recall_data = []
    support_data = []
    for selector in selector_prediction_scores_by_selector.keys():
        scores = selector_prediction_scores_by_selector.get(selector)
        f1_label_data = { }
        precision_label_data = { }
        recall_label_data = { }
        support_label_data = { }
        f1_scores = []
        accuracies = []
        precisions = []
        supports = []
        recalls = []
        for score in scores:
            f1_scores.append(score.report.general.f1_score)
            accuracies.append(score.report.general.accuracy)
            precisions.append(score.report.general.precision)
            supports.append(score.report.general.support)
            recalls.append(score.report.general.recall)
            for label_score in score.report.per_label:
                add_on_dict_list(f1_label_data, label_score.label, label_score.f1_score)
                add_on_dict_list(precision_label_data, label_score.label, label_score.precision)
                add_on_dict_list(recall_label_data, label_score.label, label_score.recall)
                add_on_dict_list(support_label_data, label_score.label, label_score.support)
        _write_dict_to_csv_file(f1_label_data, ["f1", "values"], f'raw-label-f1-scores-{selector}')
        _write_dict_to_csv_file(precision_label_data, ["precision", "values"], f'raw-label-precision-scores-{selector}')
        _write_dict_to_csv_file(recall_label_data, ["recall", "values"], f'raw-label-recall-scores-{selector}')
        _write_dict_to_csv_file(support_label_data, ["support", "values"], f'raw-label-support-{selector}')
        f1_data.append([selector, f1_scores])
        accuracy_data.append([selector, accuracies])
        precision_data.append([selector, precisions])
        recall_data.append([selector, supports])
        support_data.append([selector, recalls])
    f1_data.insert(0, ["selector", "f1 scores"])
    accuracy_data.insert(0, ["selector", "accuracy scores"])
    precision_data.insert(0, ["selector", "precision scores"])
    recall_data.insert(0, ["selector", "recall scores"])
    support_data.insert(0, ["selector", "support scores"])
    _write_to_csv_file(f1_data, f'raw-general-f1-scores')
    _write_to_csv_file(accuracy_data, f'raw-general-accuracy-scores')
    _write_to_csv_file(precision_data, f'raw-general-precision-scores')
    _write_to_csv_file(recall_data, f'raw-general-recall-scores')
    _write_to_csv_file(support_data, f'raw-general-support')
