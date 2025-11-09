import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List
from tabulate import tabulate
from src.config.general_config import CREATE_HEATMAP_BASED_ON_INFORMATIVE_FEATURES, FEATURES_TO_DISPLAY_ON_GENERAL_HEATMAP, FEATURES_TO_DISPLAY_PLUS_INFORMATIVE_ON_HEADMAP, INFORMATIVE_FEATURES_STEP_ON_CHART, SHOULD_CALCULATE_METRICS_BY_LABEL, OUTPUT_PATH, INFORMATIVE_FEATURES_OUTPUT_SUB_PATH, MAX_INFORMATIVE_FEATURES_CHART_RANGE
from src.history.ExecutionHistory import ExecutionHistory
from src.evaluation.informative_features.InformativeFeaturesScore import InformativeFeaturesScore
from src.selector.BaseWeightSelectorWrapper import BaseWeightSelectorWrapper
from src.selector.enum.SelectionMode import SelectionMode
from src.selector.enum.SelectionSpecificity import SelectionSpecificity
from src.util.feature_selection_util import get_n_features_from_rank
from src.util.numpy_util import get_interception_len, normalize
from src.util.dict_util import add_on_dict_list
from src.model.Dataset import Dataset
from src.util.performance_util import ExecutionTimeCounter
from src.util.print_util import print_with_time


def calculate_informative_features_scores(history: ExecutionHistory, dataset: Dataset) -> List[InformativeFeaturesScore]:
    '''
    Given a dataset and the execution history of a selector calculates the percentage of informative features that were selected.
    Returns an empty list when the dataset does not have well defined informative features.
    '''
    print_with_time("Calculating informative feature selection...")
    time_counter = ExecutionTimeCounter().start()
    informative_features = dataset.get_informative_features()
    informative_features_by_label = dataset.get_informative_features_per_label()
    # Returns an empty list when no informative feature was defined for the dataset
    if len(informative_features) == 0:
        return []
    informative_features_by_label_are_available_on_dataset = len(informative_features_by_label) != 0
    scores = []
    # Calculates the percentege of informative features for different selection sizes
    limit_range = MAX_INFORMATIVE_FEATURES_CHART_RANGE # dataset.get_n_features() # len(informative_features) * 5
    for selection_size in range(1, limit_range):
        # Calculates using general ranking
        if SelectionSpecificity.GENERAL in src.history.get_selection_specificities():
            scores.append(_calculate_informative_features_metrics_general(history, selection_size, informative_features))
        # Calculates using per class ranking when available
        if SHOULD_CALCULATE_METRICS_BY_LABEL and informative_features_by_label_are_available_on_dataset and SelectionSpecificity.PER_LABEL in src.history.get_selection_specificities():
            scores.extend(_calculate_informative_features_metrics_per_class(history, selection_size, informative_features_by_label))
    time_counter.print_end("Informative feature selection scores calculated for selector")
    # Returns the list of scores
    return scores

def create_informative_features_scores_output(informative_scores_by_selector: dict[str, List[InformativeFeaturesScore]]):
    _create_informative_features_chart_output_by_selector(informative_scores_by_selector)
    _create_informative_features_chart_output_by_label(informative_scores_by_selector)
    _create_informative_features_table_output(informative_scores_by_selector)

def create_heatmap(selector: BaseWeightSelectorWrapper, dataset: Dataset):
    if src.selector.get_selection_mode() == SelectionMode.WEIGHT:
        knows_informative_features = len(dataset.get_informative_features()) > 0
        if CREATE_HEATMAP_BASED_ON_INFORMATIVE_FEATURES and knows_informative_features:
            heatmap = _get_heatmap_based_on_informative_features(selector, dataset)
            _persist_heatmap(heatmap, src.selector.get_class_name())
        else:
            if SelectionSpecificity.GENERAL in src.selector.get_selection_specificities():
                ranking = src.selector.get_general_ranking()
                heatmap = _get_heatmap_based_on_feature_importance_ranking(ranking, selector, dataset)
                _persist_heatmap(heatmap, f'{src.selector.get_class_name()} - General')
            if SelectionSpecificity.PER_LABEL in src.selector.get_selection_specificities():
                ranking_per_class = src.selector.get_ranking_per_class()
                for label in range(0, dataset.get_n_labels()):
                    ranking = ranking_per_class[label]
                    heatmap = _get_heatmap_based_on_feature_importance_ranking(ranking, selector, dataset)
                    _persist_heatmap(heatmap, f'{src.selector.get_class_name()} - Label {label}')
            
def _persist_heatmap(heatmap, title: str):
    fig_size = (heatmap.shape[1] * 2.5, heatmap.shape[0] * 0.5)
    plt.figure(figsize=fig_size)
    sns.heatmap(heatmap, annot=True, fmt=".3f", annot_kws={"fontsize":16})
    plt.xlabel('Class', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.savefig(f'{OUTPUT_PATH}/{INFORMATIVE_FEATURES_OUTPUT_SUB_PATH}/heatmap-{title}.pdf', dpi=200, format='pdf', bbox_inches='tight')
    plt.close()

def _get_heatmap_based_on_informative_features(selector: BaseWeightSelectorWrapper, dataset: Dataset):
    weights_per_class = src.selector.get_weights_per_class() if SelectionSpecificity.PER_LABEL in src.selector.get_selection_specificities() else []
    if SelectionSpecificity.GENERAL in src.selector.get_selection_specificities():
        weights_general = src.selector.get_general_weights()
        weights_per_class.append(weights_general)
    weights_per_class = np.array(weights_per_class)
    headmap = []
    features_limit = len(dataset.get_informative_features()) + FEATURES_TO_DISPLAY_PLUS_INFORMATIVE_ON_HEADMAP
    for current_class in range(0, len(weights_per_class)):
        weights_per_class[current_class] = normalize(weights_per_class[current_class])
    for current_feature in range(0, features_limit):
        weights = weights_per_class[:, current_feature]
        headmap.append(weights)
    columns = list(range(0, dataset.get_n_labels())) if SelectionSpecificity.PER_LABEL in src.selector.get_selection_specificities() else []
    if SelectionSpecificity.GENERAL in src.selector.get_selection_specificities():
        columns.append('general')
    return pd.DataFrame(headmap, columns=columns, index=dataset.get_feature_names()[0:features_limit])

def _get_heatmap_based_on_feature_importance_ranking(ranking: List[int], selector: BaseWeightSelectorWrapper, dataset: Dataset):
    weights_per_class = src.selector.get_weights_per_class() if SelectionSpecificity.PER_LABEL in src.selector.get_selection_specificities() else []
    if SelectionSpecificity.GENERAL in src.selector.get_selection_specificities():
        weights_general = src.selector.get_general_weights()
        weights_per_class.append(weights_general)
    weights_per_class = np.array(weights_per_class)
    headmap = []
    headmap_index = []
    for current_class in range(0, len(weights_per_class)):
        weights_per_class[current_class] = normalize(weights_per_class[current_class])
    for ranking_index in range(0, FEATURES_TO_DISPLAY_ON_GENERAL_HEATMAP):
        feature = ranking[ranking_index]
        weights = weights_per_class[:, feature]
        headmap.append(weights)
        headmap_index.append(dataset.get_feature_names()[feature])
    columns = list(range(0, dataset.get_n_labels())) if SelectionSpecificity.PER_LABEL in src.selector.get_selection_specificities() else []
    if SelectionSpecificity.GENERAL in src.selector.get_selection_specificities():
        columns.append('general')
    return pd.DataFrame(headmap, columns=columns, index=headmap_index)

def _create_informative_features_chart_output_by_selector(informative_scores_by_selector: dict[str, List[InformativeFeaturesScore]]):
    '''
    Persist a chart per label using the previously calculated informative features percentages
    '''
    time_counter = ExecutionTimeCounter().print_start("Persisting informative features chart by src.selector...")
    informative_scores_by_selector_and_label: dict[str, List[InformativeFeaturesScore]] = {}
    for selector in informative_scores_by_src.selector.keys():
        scores = informative_scores_by_selector[selector]
        for score in scores:
            add_on_dict_list(informative_scores_by_selector_and_label, f'{selector}-{score.get_label()}', score)

    for selector_and_label in informative_scores_by_selector_and_label.keys():
        data = {
            'NumberOfSelectedFeatures': [],
            'Label': [],
            'Percentage': []
        }
        selector_label_scores = informative_scores_by_selector_and_label[selector_and_label]
        for score in selector_label_scores:
            data['NumberOfSelectedFeatures'].append(score.get_selection_size())
            data['Label'].append(f'{score.get_label()} - PIFS')
            data['Percentage'].append(score.get_percentage_of_informative_features_selected())
            data['NumberOfSelectedFeatures'].append(score.get_selection_size())
            data['Label'].append(f'{score.get_label()} - PSFI')
            data['Percentage'].append(score.get_percentage_of_selected_features_that_are_informative())
        _persist_to_file(data, selector_and_label, column="Label")

    time_counter.print_end("Informative feature selection scores persisted")

def _create_informative_features_chart_output_by_label(informative_scores_by_selector: dict[str, List[InformativeFeaturesScore]]):
    '''
    Persist a chart per label using the previously calculated informative features percentages
    '''
    time_counter = ExecutionTimeCounter().print_start("Persisting informative features chart by label...")
    informative_scores_by_label = _get_informative_features_by_label(informative_scores_by_selector)
    for label in informative_scores_by_label.keys():
        data = {
            'NumberOfSelectedFeatures': [],
            'Algorithm': [],
            'Percentage': []
        }
        scores = informative_scores_by_label[label]
        for score in scores:
            data['NumberOfSelectedFeatures'].append(score.get_selection_size())
            data['Algorithm'].append(f'{score.get_selector_name()} - PIFS')
            data['Percentage'].append(score.get_percentage_of_informative_features_selected())
            data['NumberOfSelectedFeatures'].append(score.get_selection_size())
            data['Algorithm'].append(f'{score.get_selector_name()} - PSFI')
            data['Percentage'].append(score.get_percentage_of_selected_features_that_are_informative())
        _persist_to_file(data, label)

    time_counter.print_end("Informative feature selection scores persisted")

def _create_informative_features_table_output(informative_scores_by_selector: dict[str, List[InformativeFeaturesScore]]):
    '''
    Persist a table using the previously calculated informative features percentages
    '''
    time_counter = ExecutionTimeCounter().print_start("Persisting informative features table...")
    informative_scores_by_label = _get_informative_features_by_label(informative_scores_by_selector)
    for label in informative_scores_by_label.keys():
        data = []
        scores = informative_scores_by_label[label]
        for score in scores:
            data.append([score.get_selector_name(), score.get_label(), score.get_selection_size(), score.get_percentage_of_informative_features_selected(), score.get_percentage_of_selected_features_that_are_informative()])
        # Sort by algorithm
        data = sorted(data, key=lambda x: x[0], reverse=False)
        # Insert header
        data.insert(0, ["Algorithm", "Label", "Number of features", "PIFS", "PSFI"])
        # Print tables on console
        # print(f'\n{tabulate(data, headers="firstrow", tablefmt="simple")}')
        # Define output path
        output_path = f"{OUTPUT_PATH}/{INFORMATIVE_FEATURES_OUTPUT_SUB_PATH}"
        # Persist using LaTEX format
        with open(f"{output_path}/informative-features-percentages-label-{label}.txt", "w") as file:
            file.write(tabulate(data, headers="firstrow", tablefmt="latex"))
        # Persist using HTML format
        with open(f"{output_path}/informative-features-percentages-label-{label}.txt", "w") as file:
            file.write(tabulate(data, headers="firstrow", tablefmt="html"))
        # Persist using CSV format
        with open(f'{output_path}/general-selector-prediction-average-label-{label}.csv', "w", newline="") as file:
            csvwriter = csv.writer(file)
            csvwriter.writerow(data[0])  
            csvwriter.writerows(data[1:])
        with open(f"{output_path}/by-label-selector-prediction-average-label-{label}.csv", "w", newline="") as file:
            csvwriter = csv.writer(file)
            csvwriter.writerow(data[0])  
            csvwriter.writerows(data[1:])
    time_counter.print_end("Informative feature selection scores persisted")

def _get_informative_features_by_label(informative_scores_by_selector: dict[str, List[InformativeFeaturesScore]]):
    informative_scores_by_label: dict[str, List[InformativeFeaturesScore]] = {}
    for selector in informative_scores_by_src.selector.keys():
        scores = informative_scores_by_selector[selector]
        for score in scores:
            add_on_dict_list(informative_scores_by_label, score.get_label(), score)
    return informative_scores_by_label

def _persist_to_file(data: dict[str, list], title: str, column: str = "Algorithm"):
    '''
    Persist the data as a chart into a file
    '''
    if len(data['NumberOfSelectedFeatures']) == 0:
        return
    max_x = max(data["NumberOfSelectedFeatures"])
    df = pd.DataFrame(data)
    df_pivot = df.pivot(index="NumberOfSelectedFeatures", columns=column, values="Percentage")
    # Create plot
    sns.lineplot(data=df_pivot)
    # Set figure title and legends
    plt.title('')
    plt.xlabel("Number of features")
    plt.ylabel("Percentage")
    # Show values increasing 0.1 by 0.1 on Y axis
    plt.yticks(np.arange(0.0, 1.05, 0.1))
    # Show values increasing on X axis
    plt.xticks(np.arange(0, max_x + 1, INFORMATIVE_FEATURES_STEP_ON_CHART))
    # Set legend place
    plt.legend()
    # Enable grid
    plt.grid(True)
    # Persist chart as image
    plt.savefig(f'{OUTPUT_PATH}/{INFORMATIVE_FEATURES_OUTPUT_SUB_PATH}/comparison-{title}.pdf', dpi=200, format='pdf', bbox_inches='tight')
    # Reset plt
    plt.close()

def _calculate_informative_features_metrics_general(history: ExecutionHistory, selection_size: int, informative_features: list[int]) -> InformativeFeaturesScore:
    '''
    Calculates the amount of informative features that were selected without considering the labels
    '''
    total_amount_of_informative_features = 0
    for item in src.history.get_items():
        rank = get_n_features_from_rank(item.get_general_ranking(), selection_size)
        total_amount_of_informative_features += get_interception_len(rank, informative_features)
    average_of_informative_features = total_amount_of_informative_features / len(src.history.get_items())
    return InformativeFeaturesScore(
        selector_name=src.history.get_selector_name(),
        selection_size=selection_size,
        percentage_of_informative_features_selected=average_of_informative_features / len(informative_features),
        percentage_of_selected_features_that_are_informative=average_of_informative_features / selection_size
    )

def _calculate_informative_features_metrics_per_class(history: ExecutionHistory, selection_size: int, informative_features_by_label: dict[int, list[int]]) -> list[InformativeFeaturesScore]:
    '''
    Calculates the amount of informative features that were selected for each label
    '''
    scores = []
    informative_features_amount_by_label = {}
    for label in range(0, src.history.get_n_labels()):
        informative_features_amount_by_label[label] = 0
    for item in src.history.get_items():
        rank_per_class = item.get_rank_per_class()
        for label, rank in enumerate(rank_per_class):
            informative_features = informative_features_by_label[label]
            rank = get_n_features_from_rank(rank, selection_size)
            informative_features_amount_by_label[label] += get_interception_len(rank, informative_features)
    for label in range(0, src.history.get_n_labels()):
        informative_features = informative_features_by_label[label]
        average = informative_features_amount_by_label[label] / len(src.history.get_items())
        score = InformativeFeaturesScore(
            selector_name=src.history.get_selector_name(),
            selection_size=selection_size,
            percentage_of_informative_features_selected=average / len(informative_features),
            percentage_of_selected_features_that_are_informative=average / selection_size,
            label=label
        )
        scores.append(score)
    return scores