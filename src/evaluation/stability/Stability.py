import csv
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import List
from tabulate import tabulate

from src.model.SelectorSpecificity import SelectorSpecificity
from src.model.SelectorType import SelectorType
from src.domain.selector.types.base.BaseSelector import BaseSelector
from src.config.general_config import MAX_FEATURES_TO_DISPLAY_ON_SELECTION_STABILITY_CHART, STABILITY_INITIAL_END, STABILITY_INITIAL_STEP, STABILITY_LIMIT, STABILITY_SHOULD_CREATE_INDIVIDUAL_CHARTS_FOR_EACH_SELECTION_SIZE, SHOULD_CALCULATE_METRICS_BY_LABEL, OUTPUT_PATH, STABILITY_OUTPUT_SUB_PATH, STABILITY_STEP, STABILITY_STEP_ON_EVOLUTION_CHART
from src.history.ExecutionHistory import ExecutionHistory
from src.config.stability_metrics_config import STABILITY_METRICS_TYPES
from src.evaluation.stability.StabilityScore import StabilityScore
from src.util.dict_util import add_on_dict_list
from src.util.matrix_util import sort_matrix_columns
from src.util.performance_util import ExecutionTimeCounter
from src.util.print_util import print_load_bar, print_with_time


def generate_feature_selection_stability_chart(best_selector: BaseSelector, history: ExecutionHistory, feature_names: List[str]) -> None:
    '''
    Given the history at least one item generate bump charts to compare the general and by label feature selections when available.
    Model should have feature selection by rank.
    '''
    if len(history.get_items()) == 0:
        return
    if SelectorType.RANK not in history.get_available_seletion_modes():
        return
    # Define features to be displayed
    limit = min(history.get_n_features(), MAX_FEATURES_TO_DISPLAY_ON_SELECTION_STABILITY_CHART)
    features_to_display = best_selector.get_general_ranking()
    if len(features_to_display) > limit:
        features_to_display = features_to_display[0:limit]
    # Generate chart for general rank when available
    if SelectorSpecificity.GENERAL in history.get_selection_specificities():
        _generate_feature_selection_chart_for_general(history, feature_names, features_to_display)
    # Generate charts for each label rank when available
    if SelectorSpecificity.PER_LABEL in history.get_selection_specificities():
        _generate_feature_selection_chart_for_label(history, feature_names, features_to_display)

def calculate_stability_scores(history: ExecutionHistory) -> List[StabilityScore]:
    '''
    Given the execution history of a selector, calculate the stability scores related to 
    the selections done on each round
    '''
    print_with_time("Calculating stability scores...")
    time_counter = ExecutionTimeCounter().start()
    scores = []
    for Metric in STABILITY_METRICS_TYPES:
        print_with_time(f"Stability for {Metric.get_name()}")
        selection_sizes = _get_stability_sizes(history.get_n_features())
        specificities = history.get_selection_specificities()
        for i, selection_size in enumerate(selection_sizes):
            print_load_bar(i, len(selection_sizes))
            if selection_size == 1:
                continue
            metric = Metric(history, selection_size)
            if metric.should_execute():
                if SelectorSpecificity.GENERAL in specificities:
                    scores.append(metric.calculate_general(history.get_selector_name()))
                if SHOULD_CALCULATE_METRICS_BY_LABEL and SelectorSpecificity.PER_LABEL in specificities:
                    scores.extend(metric.calculate_per_class(history.get_selector_name()))
    time_counter.print_end("Stability scores calculation")
    return scores

def _get_stability_sizes(n_features: int):
    limit = STABILITY_LIMIT if STABILITY_LIMIT != None else n_features
    steps = list(range(2, STABILITY_INITIAL_END, STABILITY_INITIAL_STEP))
    last_steps = list(range(STABILITY_INITIAL_END, limit, STABILITY_STEP))
    steps.extend(last_steps)
    if steps[-1] != limit:
        steps.append(limit)
    if steps[-1] != n_features:
        steps.append(n_features)
    return steps

def create_stability_table_and_charts(stability_scores_by_selector: dict[str, List[StabilityScore]], n_features: int):
    _create_stability_table(stability_scores_by_selector)
    if STABILITY_SHOULD_CREATE_INDIVIDUAL_CHARTS_FOR_EACH_SELECTION_SIZE:
        _create_stability_charts(stability_scores_by_selector)
    _create_stability_evolution_chart(stability_scores_by_selector, n_features)

def _create_stability_table(stability_scores_by_selector: dict[str, List[StabilityScore]]):
    '''
    Given a list of stability scores of different selectors create a single table to compare all results.
    The table will be written to an output file.
    '''
    # Add rows to data list
    data = []
    for selector_name, selector_scores in stability_scores_by_selector.items():
        for score in selector_scores:
            data.append([selector_name, score.get_metric(), score.get_target_label(), score.get_selection_size(), score.get_score()])
    # Sort by selection size
    data = sorted(data, key=lambda x: x[3], reverse=True)
    # Insert header
    data.insert(0, ["Algorithm", "Stability metric", "Target label", "Amount of selected features", "Score"])
    # Print tabla on console
    #print(f'\n{tabulate(data, headers="firstrow", tablefmt="simple")}')
    # Define output path
    output_path = f"{OUTPUT_PATH}/{STABILITY_OUTPUT_SUB_PATH}"
    # Persist using LaTEX format
    with open(f"{output_path}/selector-stability.txt", "w") as file:
        file.write(tabulate(data, headers="firstrow", tablefmt="latex"))
    # Persist using HTML format
    with open(f"{output_path}/selector-stability.html", "w") as file:
        file.write(tabulate(data, headers="firstrow", tablefmt="html"))
    # Persist using CSV format
    with open(f'{output_path}/average-execution-time.csv', "w", newline="") as file:
        csvwriter = csv.writer(file)
        csvwriter.writerow(data[0])  
        csvwriter.writerows(data[1:])

def _create_stability_charts(stability_scores_by_selector: dict[str, List[StabilityScore]]):
    '''
    Given a list of stability scores of different selectors create multiple charts for each selection size 
    with scores from all selectores.
    The image will be persisted to an output file.
    '''
    selectors = list(stability_scores_by_selector.keys())
    selection_sizes = _get_stability_sizes()
    # Create a chart per metric
    for Metric in STABILITY_METRICS_TYPES:
        # Create a chart per selection size
        for selection_size in selection_sizes:
            scores_by_label: dict[str, List[float]] = { }
            # Create a group per selector
            for selector in selectors:
                # For each score from selector
                for score in stability_scores_by_selector[selector]:
                    # When the score refers to the current selection size and metric
                    if score.get_metric() == Metric.get_name() and score.get_selection_size() == selection_size:
                        # Add score to the correspondent metric list 
                        add_on_dict_list(scores_by_label, score.get_target_label(), score.get_score())
            x = np.arange(len(selectors))
            width = 1.0 / (len(scores_by_label.items()) + 1)  # the width of the bars
            multiplier = 0
            figure, ax = plt.subplots(layout='constrained')
            # Set figure dimensions
            figure.set_size_inches(24, 16)
            # Add metric bars for all selectors
            for label, scores in scores_by_label.items():
                offset = width * multiplier
                rects = ax.bar(x + offset, scores, width, label=f'Label {label}')
                ax.bar_label(rects, padding=3)
                multiplier += 1
            ax.set_ylabel('Score')
            ax.set_xlabel('Metric')
            ax.set_title(f'{Metric.get_name()} stability score by selector given a selection of {selection_size} features')
            ax.set_xticks(x + width, selectors)
            ax.legend(loc='upper left', ncols=3)
            # Persist chart as image
            plt.savefig(f'{OUTPUT_PATH}/{STABILITY_OUTPUT_SUB_PATH}/selector-stability-{Metric.get_name().lower()}-{str(selection_size)}.pdf', dpi=200, format='pdf', bbox_inches='tight')
            # Reset plt
            plt.close()   

def _create_stability_evolution_chart(stability_scores_by_selector: dict[str, List[StabilityScore]], n_features: int):
    '''
    Given a list of stability scores for multiple predictors, create a evolution chart by number of features
    '''
    # Get all scores by metric and label
    score_by_metric_and_label = {}
    for scores in stability_scores_by_selector.values():
        for score in scores:
            add_on_dict_list(score_by_metric_and_label, f'{score.get_metric()} - Label {score.get_target_label()}', score)
            
    # Create one chat per each metric and label
    for metric_and_label in score_by_metric_and_label.keys():
        scores: List[StabilityScore] = score_by_metric_and_label[metric_and_label]
        data = {
            'NumberOfFeatures': [],
            'Algorithm': [],
            'Score': []
        }
        for score in scores:
            if STABILITY_LIMIT is None or score.get_selection_size() <= STABILITY_LIMIT:             
                data['NumberOfFeatures'].append(score.get_selection_size())
                data['Algorithm'].append(score.get_selector_name())
                data['Score'].append(score.get_score())
        _persist_evolution_chart_to_file(data, metric_and_label, n_features)
    # Create one per each metric and selector
    for selector in stability_scores_by_selector.keys():
        scores = stability_scores_by_selector[selector]
        # Get all scores by metric and label
        selector_score_by_metric: dict[str, List[StabilityScore]] = {}
        for score in scores:
            add_on_dict_list(selector_score_by_metric, score.get_metric(), score)
        # Create one chart per each metric and label
        for metric_and_label in selector_score_by_metric.keys():
            scores = selector_score_by_metric[metric_and_label]
            data = {
                'NumberOfFeatures': [],
                'Metric': [],
                'Score': []
            }
            for score in scores:
                if STABILITY_LIMIT is None or score.get_selection_size() <= STABILITY_LIMIT:
                    data['NumberOfFeatures'].append(score.get_selection_size())
                    data['Metric'].append(f'{metric_and_label} - Label {score.get_target_label()}')
                    data['Score'].append(score.get_score())
            _persist_evolution_chart_to_file(data, f'{selector} - {metric_and_label}', n_features, column="Metric")

def _persist_evolution_chart_to_file(data: dict[str, list], name: str, n_features: int, column: str = "Algorithm"):
    '''
    Persist the data as a chart into a file
    '''
    if len(data['NumberOfFeatures']) == 0:
        return
    df = pd.DataFrame(data)
    df_pivot = df.pivot(index="NumberOfFeatures", columns=column, values="Score")
    # Create plot
    sns.lineplot(data=df_pivot)
    # Set figure title and legends
    plt.title('')
    plt.xlabel("Number of features")
    plt.ylabel("Score")
    # Show values increasing one by one on X axis
    #xticks = np.arange(0, STABILITY_LIMIT + 1, STABILITY_STEP_ON_EVOLUTION_CHART)
    #plt.xticks(xticks)
    # Enable grid
    plt.grid(True)
    # Persist chart as image
    plt.savefig(f'{OUTPUT_PATH}/{STABILITY_OUTPUT_SUB_PATH}/evolution-{name}.pdf', format='pdf', bbox_inches='tight')
    # Reset plt
    plt.close()

def _generate_feature_selection_chart(title: str, filename: str, ranks: List[np.ndarray], history: ExecutionHistory, feature_names: List[str], features_to_display: List[int]):
    '''
    Given a list of ranks generate a bump chart to compare the feature selections
    '''
    # Create list with all executions
    executions = np.arange(0, len(ranks), 1)
    # Create list with all possible positions
    rank_positions = np.arange(0, max(_get_stability_sizes(history.get_n_features())) + 1, STABILITY_STEP_ON_EVOLUTION_CHART)
    # Set figure dimensions
    plt.figure().set_size_inches(36, 24)
    # Create a row for each rank
    positions_matrix = []
    selected_features = []
    number_of_executions = len(ranks)
    for feature_number in features_to_display:
        amountOfTimesOnTheTop = 0
        for rank in ranks:
            position = rank.tolist().index(feature_number)
            if position < len(features_to_display):
                amountOfTimesOnTheTop += 1
        if amountOfTimesOnTheTop > (number_of_executions * 0.7):
            selected_features.append(feature_number)
    for feature_number in selected_features:
        positions = []
        for rank in ranks:
            position = rank.tolist().index(feature_number)
            if position < len(features_to_display):
                positions.append(position)
            else:
                positions.append(len(features_to_display) + 1)
        positions_matrix.append(positions)
    positions_matrix = sort_matrix_columns(positions_matrix)
    for positions in positions_matrix:
        plt.plot(executions, positions, marker='o', linewidth=3.0, markersize=10)
    # Define x and y lines to follow the positions and feature names
    plt.xticks(executions)
    plt.xlabel('Execution', fontsize=20)
    plt.yticks(np.arange(0, len(features_to_display), len(features_to_display)/5)) # Ranking of each feature
    plt.ylabel('Rank position', fontsize=20)
    # Invert the y-axis
    plt.gca().invert_yaxis()
    #plt.ylim(top=len(features_to_display), bottom=len(features_to_display))
    # Define title
    plt.title('')
    # Enable grid
    plt.grid(True)
    # Disable legend
    plt.legend().remove()
    # Increate the chunksize of Matplot lib
    mpl.rcParams['agg.path.chunksize'] = 10000
    # Persist chart as image
    plt.savefig(f'{OUTPUT_PATH}/{STABILITY_OUTPUT_SUB_PATH}/{filename}.pdf', dpi=200, format='pdf', bbox_inches='tight')
    # Reset plt
    plt.close()

def _generate_feature_selection_chart_for_general(history: ExecutionHistory, feature_names: List[str], features_to_display: List[int]):
    ranks: List[np.ndarray] = []
    # Extract rank from each training
    for item in history.get_items():
        ranks.append(item.get_general_ranking())
    # Create chart
    _generate_feature_selection_chart(
        title=f'{history.get_selector_name()} - Comparison between different trainings - General', 
        filename=f'selector-rank-comparison-{history.get_selector_name().lower()}-general', 
        ranks=ranks, 
        history=history, 
        feature_names=feature_names,
        features_to_display=features_to_display
    )

def _generate_feature_selection_chart_for_label(history: ExecutionHistory, feature_names: List[str], features_to_display: List[int]):
   # Create chart for each label
   for label in range(0, history.get_n_labels()):
        ranks: List[np.ndarray] = []
        # Extract rank from each training
        for item in history.get_items():
            ranks.append(item.get_rank_per_class()[label])
        # Create chart
        _generate_feature_selection_chart(
            title=f'{history.get_selector_name()} - Comparison between different trainings - Label {label}', 
            filename=f'selector-rank-comparison-{history.get_selector_name().lower()}-label-{label}', 
            ranks=ranks, 
            history=history, 
            feature_names=feature_names,
            features_to_display=features_to_display
        )