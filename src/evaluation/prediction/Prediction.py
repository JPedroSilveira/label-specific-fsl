import csv
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from tabulate import tabulate

from src.domain.selector.types.base.BaseSelector import BaseSelector

import src.config.predictor_types_config as predictor_types_config
from src.config.general_config import PREDICTOR_INITIAL_END, PREDICTOR_INITIAL_STEP, PREDICTOR_LIMIT, PREDICTOR_SHOULD_CREATE_INDIVIDUAL_CHARTS_FOR_EACH_SELECTION_SIZE, OUTPUT_PATH, PREDICTOR_PERFORMANCE_OUTPUT_SUB_PATH, PREDICTOR_STEP, PREDICTOR_STEP_ON_CHART, SELECTOR_PREDICTION_PERFORMANCE_OUTPUT_SUB_PATH
from src.model.Dataset import Dataset
from src.model.SplittedDataset import SplittedDataset
from src.util.dict_util import add_on_dict_list
from src.util.performance_util import ExecutionTimeCounter
from src.util.print_util import print_load_bar, print_with_time
from src.util.classification_report_util import calculate_classification_report
from src.history.ExecutionHistory import ExecutionHistory
from src.evaluation.prediction.prediction_evaluator.BasePredictionEvaluator import BasePredictionEvaluator
from src.evaluation.prediction.PredictionScore import BasePredictionScore, SelectorPredictionScore, PredictorPredictionScore
from src.evaluation.prediction.PredictionAverage import PredictorPredictionScoreStatistics, SelectorPredictionScoreStatistics, PredictionScoreAverageByLabel


def calculate_prediction_score_from_selector(selector: BaseSelector, test_dataset: Dataset) -> SelectorPredictionScore:
    '''
    Given a trained selector capable of doing preditions and a test dataset, calculate it's prediction scores
    '''
    score = None
    print_with_time("Calculating prediction score from selector")
    timer_counter = ExecutionTimeCounter().start()
    # Verify if selector is able to do predictions
    if selector.can_predict():
        y_pred = selector.predict(test_dataset)
        report = calculate_classification_report(test_dataset, y_pred)
        score = SelectorPredictionScore(test_dataset.get_n_features(), selector, report)
        # Print selector prediction F1 Score
        print_with_time(f"Selector general f1 score: {score.report.general.f1_score}")
    timer_counter.print_end("Prediction selector test")
    return score

def calculate_prediction_scores_from_feature_selection(selector: BaseSelector, splitted_dataset: SplittedDataset) -> list[PredictorPredictionScoreStatistics]:
    '''
    Given a trained selector and a dataset with train and test data, calculate the prediction scores of predictions models trained
    with the feature selections
    '''
    score_averages: List[PredictorPredictionScoreStatistics] = []
    print_with_time('Calculating predictor scores...')
    time_counter = ExecutionTimeCounter().start()
    # Calculate prediction score for each feature selection size
    selection_sizes = _get_predictor_sizes(splitted_dataset.get_n_features())
    for i, selection_size in enumerate(selection_sizes):
        print_load_bar(i + 1, len(selection_sizes))
        # Calculate the prediction score for each predictor
        for predictor_type in predictor_types_config.PREDICTOR_TYPES:
            #print_with_time(f'Running predictor {predictor_type.get_name()} for selection size {selection_size}...')
            # Calculate the prediction score for each evaluation type, 
            # between general and per label feature selections and between rank and weight selections
            for evaluator_type in predictor_types_config.PREDICTION_EVALUATOR_TYPES:
                #print_with_time(f'Running prediction evaluator {evaluator_type.get_name()}')
                scores: List[BasePredictionScore] = []
                evaluator: BasePredictionEvaluator = evaluator_type(selector, selection_size, predictor_type)
                # Verify if evaluation is able to evaluate given the specific seletor
                # Example.: A selector that only defines a rank of features can not be evaluated by a weight evaluator
                if evaluator.should_execute(selector):
                    # Executes the evaluator, that you will a list of scores based on multiple trainings
                    scores = evaluator.calculate(selector, splitted_dataset)
                if len(scores) > 0:
                    score_average: PredictorPredictionScoreStatistics = _calculate_prediction_average(selector.get_n_labels(), selector.get_class_name(), scores)
                    score_average.predictor_name = predictor_type.get_name()
                    score_average.evaluator_name = evaluator_type.get_name()
                    score_average.selection_size = selection_size
                    score_averages.append(score_average)
    time_counter.print_end("Predictor scores test")
    return score_averages

def create_selectors_prediction_average_table_and_chart(selector_prediction_score_average_by_selector: dict[str, SelectorPredictionScoreStatistics]):
    if not selector_prediction_score_average_by_selector:
        return
    _create_selectors_prediction_average_table(selector_prediction_score_average_by_selector)
    _create_selectors_prediction_average_chart(selector_prediction_score_average_by_selector)
    
def calculate_prediction_average_from_selector(history: ExecutionHistory):
    '''
    Given a list of prediction scores for a selector, calculates the prediction scores average
    '''
    print_with_time("Calculating prediction score averages from src.selector...")
    time_counter = ExecutionTimeCounter().start()
    scores = list(map(lambda item: item.get_prediction_score(), history.get_items()))
    average = _calculate_prediction_average(history.get_n_labels(), history.get_selector_name(), scores)
    time_counter.print_end("Selector prediction score average calculation")
    return average

def create_predictors_table_and_chart(predictors_scores_by_selector: dict[str, list[PredictorPredictionScoreStatistics]], n_features: int):
    _create_predictors_prediction_table(predictors_scores_by_selector)
    if PREDICTOR_SHOULD_CREATE_INDIVIDUAL_CHARTS_FOR_EACH_SELECTION_SIZE:
        _create_predictiors_prediction_chart(predictors_scores_by_selector, n_features)
    _create_prediction_evolution_chart(predictors_scores_by_selector)
    _create_prediction_evolution_chart_with_errorbar(predictors_scores_by_selector)

def create_selectors_prediction_chart(selector_prediction_scores_by_selector: dict[str, List[SelectorPredictionScore]]):
    '''
    Given a dictionary with the prediction scores for each selector, persist a boxplot chart
    '''
    # Extract data for plotting
    selector_prediction_f1_scores_by_selector: dict[str, List[float]] = {}
    for selector in selector_prediction_scores_by_selector.keys():
        scores = selector_prediction_scores_by_selector[selector]
        for score in scores:
            add_on_dict_list(selector_prediction_f1_scores_by_selector, selector, score.report.general.f1_score)
    data = [selector_prediction_f1_scores_by_selector[label] for label in selector_prediction_f1_scores_by_selector]
    labels = list(selector_prediction_scores_by_selector.keys())
    # Set figure dimensions
    plt.figure()
    # Create the boxplot
    plt.boxplot(data)
    # Set figure title and legends
    plt.title('')
    plt.ylabel('F1 Score (test)')
    #plt.xlabel('Feature selection algorithm')
    plt.xticks(range(1, len(labels) + 1), labels)
    # Persist chart as image
    plt.savefig(f'{OUTPUT_PATH}/{SELECTOR_PREDICTION_PERFORMANCE_OUTPUT_SUB_PATH}/selectors-prediction-boxplot.pdf', format='pdf', bbox_inches='tight')
    # Reset plt
    plt.close()

def _get_predictor_sizes(n_features: int):
    limit = PREDICTOR_LIMIT if PREDICTOR_LIMIT != None else n_features
    steps = list(range(1, PREDICTOR_INITIAL_END, PREDICTOR_INITIAL_STEP))
    last_steps = list(range(PREDICTOR_INITIAL_END, limit, PREDICTOR_STEP))
    steps.extend(last_steps)
    if steps[-1] != limit:
        steps.append(limit)
    return steps

def _create_prediction_evolution_chart_with_errorbar(predictors_scores_by_selector: dict[str, List[PredictorPredictionScoreStatistics]]):
    for selector in predictors_scores_by_selector.keys():
        selector_scores = predictors_scores_by_selector.get(selector)
        scores_by_predictor: dict[str, List[PredictorPredictionScoreStatistics]] = {}
        for score in selector_scores:
            add_on_dict_list(scores_by_predictor, score.predictor_name, score)
        for predictor in scores_by_predictor.keys():
            predictor_scores = scores_by_predictor.get(predictor)
            scores_by_evaluator: dict[str, List[PredictorPredictionScoreStatistics]] = {}
            for score in predictor_scores:
                add_on_dict_list(scores_by_evaluator, score.evaluator_name, score)
            for evaluator in scores_by_evaluator.keys():
                evaluator_scores = scores_by_evaluator.get(evaluator)
                selected_features = []
                avg_scores = []
                std_devs = []
                for score in evaluator_scores:
                    selected_features.append(score.selection_size)
                    avg_scores.append(score.f1_score_avg)
                    std_devs.append(score.f1_score_var)
                plt.errorbar(selected_features, avg_scores, yerr=std_devs, fmt='o-', capsize=5)
                plt.xlabel('Number of features')
                plt.ylabel('F1 score (test)')
                plt.title('')
                # Display the plot
                plt.grid(True)
                # Persist chart as image
                plt.savefig(f'{OUTPUT_PATH}/{PREDICTOR_PERFORMANCE_OUTPUT_SUB_PATH}/evolution-{selector}-{predictor}-{evaluator}-with-error-bar.pdf', dpi=200, format='pdf', bbox_inches='tight')
                # Reset plt
                plt.close()
            
def _create_selectors_prediction_average_chart(selector_prediction_score_average_by_selector: dict[str, SelectorPredictionScoreStatistics]):
    selectors = list(selector_prediction_score_average_by_selector.keys())
    label_scores: dict[str, Tuple[int]] = { }
    for score in selector_prediction_score_average_by_selector.values():
        for score_by_label in score.by_label:
            add_on_dict_list(label_scores, score_by_label.label, (round(score_by_label.f1_score_avg, 2), round(score_by_label.f1_score_var, 2)))
    x = np.arange(len(selectors))  # the label locations
    width = 0.2  # the width of the bars
    multiplier = 0
    _, ax = plt.subplots(layout='constrained')
    # Set figure dimensions
    for attribute, values in label_scores.items():
        offset = width * multiplier
        averages = [value[0] for value in values]
        variances = [value[1] for value in values]
        rects = ax.bar(x + offset, averages, width, label=f'Label {attribute}', yerr=variances)
        ax.bar_label(rects, label_type='center')
        ax.errorbar(x + offset, averages, yerr=variances, color="r")
        multiplier += 1
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('F1 score (test)')
    ax.set_title('')
    ax.set_xticks(x + width, selectors)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_ylim(0, 1.0)
    # Persist chart as image
    plt.savefig(f'{OUTPUT_PATH}/{SELECTOR_PREDICTION_PERFORMANCE_OUTPUT_SUB_PATH}/selectors-prediction-comparison.pdf', format='pdf', bbox_inches='tight')
    # Reset plt
    plt.close()   

def _create_selectors_prediction_average_table(selector_prediction_score_average_by_selector: dict[str, SelectorPredictionScoreStatistics]):
    '''
    Given a list of predictions of different selectors create tables to compare general predictions and predictions by label.
    Tables are written to output files.
    '''
    # Add rows to general list and by label list
    general_data = []
    by_label_data = []
    for score in selector_prediction_score_average_by_selector.values():
        general_data.append([score.selector_name, f'{round(score.accuracy_avg, 5)};{round(score.accuracy_var, 5)}', f'{round(score.precision_avg, 5)};{round(score.precision_var, 5)}', f'{round(score.recall_avg, 5)};{round(score.recall_var, 5)}', f'{round(score.f1_score_avg, 5)};{round(score.f1_score_var, 5)}'])
        for score_by_label in score.by_label:
            by_label_data.append([score.selector_name, score_by_label.label, f'{round(score_by_label.precision_avg, 5)};{round(score_by_label.precision_var, 5)}', f'{round(score_by_label.recall_avg, 5)};{round(score_by_label.recall_var, 5)}', f'{round(score_by_label.f1_score_avg, 5)};{round(score_by_label.f1_score_var, 5)}'])
    # Sort by label list using the label column
    by_label_data = sorted(by_label_data, key=lambda x: x[1], reverse=True)
    # Insert header
    general_data.insert(0, ["Algorithm", "Accuracy", "Precision", "Recall", "F1 score"])
    by_label_data.insert(0, ["Algorithm", "Label", "Precision", "Recall", "F1 score"])
    # Print tables on console
    #print(f'\n{tabulate(general_data, headers="firstrow", tablefmt="simple")}')
    #print(f'\n{tabulate(by_label_data, headers="firstrow", tablefmt="simple")}')
    # Define output path
    output_path = f"{OUTPUT_PATH}/{SELECTOR_PREDICTION_PERFORMANCE_OUTPUT_SUB_PATH}"
    # Persist using LaTEX format
    with open(f"{output_path}/general-selector-prediction-average.txt", "w") as file:
        file.write(tabulate(general_data, headers="firstrow", tablefmt="latex"))
    with open(f"{output_path}/by-label-selector-prediction-average.txt", "w") as file:
        file.write(tabulate(by_label_data, headers="firstrow", tablefmt="latex"))
    # Persist using HTML format
    with open(f"{output_path}/general-selector-prediction-average.html", "w") as file:
        file.write(tabulate(general_data, headers="firstrow", tablefmt="html"))
    with open(f"{output_path}/by-label-selector-prediction-average.html", "w") as file:
        file.write(tabulate(by_label_data, headers="firstrow", tablefmt="html"))
    # Persist using CSV format
    with open(f'{output_path}/general-selector-prediction-average.csv', "w", newline="") as file:
        csvwriter = csv.writer(file)
        csvwriter.writerow(general_data[0])  
        csvwriter.writerows(general_data[1:])
    with open(f"{output_path}/by-label-selector-prediction-average.csv", "w", newline="") as file:
        csvwriter = csv.writer(file)
        csvwriter.writerow(by_label_data[0])  
        csvwriter.writerows(by_label_data[1:])

def _create_prediction_evolution_chart(predictors_scores_by_selector: dict[str, List[PredictorPredictionScoreStatistics]]):
    '''
    Given a list of prediction scores for a selector with multiple predictors, create a evolution chart by number of features
    '''
    # Get all scores by predictor
    scores_by_predictor: dict[str, List[PredictorPredictionScoreStatistics]] = {}
    for scores in predictors_scores_by_selector.values():
        for score in scores:
            add_on_dict_list(scores_by_predictor, score.predictor_name, score)
    # Create one per each predictor
    for predictor in scores_by_predictor.keys():
        data = {
            'NumberOfFeatures': [],
            'Algorithm': [],
            'F1 score': []
        }
        scores = scores_by_predictor[predictor]
        for score in scores:
            data['NumberOfFeatures'].append(score.selection_size)
            data['Algorithm'].append(f'{score.selector_name} - {score.evaluator_name}')
            data['F1 score'].append(score.f1_score_avg)
        _persist_evolution_chart_to_file(data, f'predictor-{predictor}')
    # Create one per each selector
    for selector in predictors_scores_by_selector.keys():
        data = {
            'NumberOfFeatures': [],
            'Algorithm': [],
            'F1 score': []
        }
        scores = predictors_scores_by_selector[selector]
        for score in scores:
            data['NumberOfFeatures'].append(score.selection_size)
            data['Algorithm'].append(f'{score.predictor_name} - {score.evaluator_name}')
            data['F1 score'].append(score.f1_score_avg)
        _persist_evolution_chart_to_file(data, f'selector-{selector}')

def _persist_evolution_chart_to_file(data: dict[str, list], name: str):
    '''
    Persist the data as a chart into a file
    '''
    max_number_of_selected_features = max(data['NumberOfFeatures'])
    if len(data['NumberOfFeatures']) == 0:
        return
    df = pd.DataFrame(data)
    df_pivot = df.pivot(index="NumberOfFeatures", columns="Algorithm", values="F1 score")
    # Create plot
    sns.lineplot(data=df_pivot)
    # Set figure title and legends
    plt.title('')
    plt.xlabel("Number of features")
    plt.ylabel("F1 score (test)")
    # Show values increasing 0.1 by 0.1 on Y axis
    plt.yticks(np.arange(0.0, 1.05, 0.1))
    # Set legend place outside the graph
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # Enable grid
    plt.grid(True)
    # Show values increasing one by one on X axis
    plt.xticks(np.arange(0, max_number_of_selected_features + 1, PREDICTOR_STEP_ON_CHART))
    # Persist chart as image
    plt.savefig(f'{OUTPUT_PATH}/{PREDICTOR_PERFORMANCE_OUTPUT_SUB_PATH}/evolution-{name}.pdf', dpi=200, format='pdf', bbox_inches='tight')
    # Reset plt
    plt.close()

def _create_predictiors_prediction_chart(predictors_scores_by_selector: dict[str, list[PredictorPredictionScoreStatistics]], n_features: int):
    selectors = list(predictors_scores_by_selector.keys())
    selection_sizes = _get_predictor_sizes(n_features)
    # Generate a chart for each predictor type
    for predictor_type in predictor_types_config.PREDICTOR_TYPES:
        # Generate a chart for each evaluator type
        for evaluator_type in predictor_types_config.PREDICTION_EVALUATOR_TYPES:
            evaluator_name = evaluator_type.get_name()
            predictor_name = predictor_type.get_name()
            # Generate a chart for each selection size
            for selection_size in selection_sizes:
                label_scores = { }
                for scores in predictors_scores_by_selector.values():
                    for score in scores:
                        if score.selection_size != selection_size or score.evaluator_name != evaluator_name or score.predictor_name != predictor_name:
                            continue
                        for score_by_label in score.by_label:
                            add_on_dict_list(label_scores, score_by_label.label, score_by_label.f1_score_avg)
                x = np.arange(len(selectors))  # the label locations
                width = 1.2 / (len(label_scores.items()) + 1)  # the width of the bars
                multiplier = 0
                fig, ax = plt.subplots(layout='constrained')
                # Set figure dimensions
                fig.set_size_inches(24, 16)
                for attribute, measurement in label_scores.items():
                    offset = width * multiplier
                    rects = ax.bar(x + offset, measurement, width, label=attribute)
                    ax.bar_label(rects, padding=3)
                    multiplier += 1
                # Add some text for labels, title and custom x-axis tick labels, etc.
                ax.set_ylabel('F1 score (test)')
                #ax.set_title(f'F1 score comparison with {selection_size} features and model {predictor_name}', pad=20)
                ax.set_title('')
                ax.set_xticks(x + width, selectors)
                ax.set_yticks(np.arange(0, 1, 0.05))
                ax.legend(loc='upper left', ncols=3)
                # Persist chart as image
                plt.savefig(f'{OUTPUT_PATH}/{PREDICTOR_PERFORMANCE_OUTPUT_SUB_PATH}/predictor-{predictor_name.lower()}-prediction-selection-{selection_size}-evaluator-{evaluator_name.lower().replace(" ", "-")}.pdf', dpi=200, format='pdf', bbox_inches='tight')
                # Reset plt
                plt.close()   

def _create_predictors_prediction_table(predictors_scores_by_selector: dict[str, list[PredictorPredictionScoreStatistics]]):
    '''
    Given a list of predictions of different predictors create tables to compare general predictions and predictions by label.
    Tables are written to output files.
    '''
    # Add rows to general list and by label list
    general_data = []
    by_label_data = []
    for scores in predictors_scores_by_selector.values():
        for score in scores:
            general_data.append([score.predictor_name, score.selector_name, score.evaluator_name, score.selection_size, f'{round(score.accuracy_avg, 5)};{round(score.accuracy_var, 5)}', f'{round(score.precision_avg, 5)};{round(score.precision_var, 5)}', f'{round(score.recall_avg, 5)};{round(score.recall_var, 5)}', f'{round(score.f1_score_avg, 5)};{round(score.f1_score_var, 5)}'])
            for score_by_label in score.by_label:
                by_label_data.append([score.predictor_name, score.selector_name, score.evaluator_name, score.selection_size, score_by_label.label, f'{round(score_by_label.precision_avg, 5)};{round(score_by_label.precision_var, 5)}' , f'{round(score_by_label.recall_avg, 5)};{round(score_by_label.recall_var, 5)}', f'{round(score_by_label.f1_score_avg, 5)};{round(score_by_label.f1_score_var, 5)}'])
    # Sort by label list using the label column
    by_label_data = sorted(by_label_data, key=lambda x: x[3], reverse=True)
    # Insert header
    general_data.insert(0, ["Predictor", "Algorithm", "Evaluation type", "Amount of selected features", "Accuracy", "Precision", "Recall", "F1 score"])
    by_label_data.insert(0, ["Predictor", "Algorithm", "Evaluation type", "Amount of selected features", "Label", "Precision", "Recall", "F1 score"])
    # Print tables on console
    #print(f'\n{tabulate(general_data, headers="firstrow", tablefmt="simple")}')
    #print(f'\n{tabulate(by_label_data, headers="firstrow", tablefmt="simple")}')
    # Define output path
    output_path = f"{OUTPUT_PATH}/{PREDICTOR_PERFORMANCE_OUTPUT_SUB_PATH}"
    # Persist using LaTEX format
    with open(f"{output_path}/general-predictor-prediction.txt", "w") as file:
        file.write(tabulate(general_data, headers="firstrow", tablefmt="latex"))
    with open(f"{output_path}/by-label-predictor-prediction.txt", "w") as file:
        file.write(tabulate(by_label_data, headers="firstrow", tablefmt="latex"))
    # Persist using HTML format
    with open(f"{output_path}/general-predictor-prediction.html", "w") as file:
        file.write(tabulate(general_data, headers="firstrow", tablefmt="html"))
    with open(f"{output_path}/by-label-predictor-prediction.html", "w") as file:
        file.write(tabulate(by_label_data, headers="firstrow", tablefmt="html"))
    # Persist using CSV format
    with open(f"{output_path}/general-predictor-prediction.csv", "w", newline="") as file:
        csvwriter = csv.writer(file)
        csvwriter.writerow(general_data[0])  
        csvwriter.writerows(general_data[1:])
    with open(f"{output_path}/by-label-predictor-prediction.csv", "w", newline="") as file:
        csvwriter = csv.writer(file)
        csvwriter.writerow(by_label_data[0])  
        csvwriter.writerows(by_label_data[1:])

def _calculate_prediction_average(n_labels: int, selector_name: str, scores: List[SelectorPredictionScore | PredictorPredictionScore]) -> SelectorPredictionScoreStatistics | PredictorPredictionScoreStatistics:
    '''
    Given a list of prediction scores, calculate the prediction average
    '''
    # Verify if list of scores is not empty
    if len(scores) == 0:
        raise ValueError("List of scores from history can not be empty")
    
    # Extract the first score to get selection informations
    sample_score = scores[0]
    
    # Create the average prediction model based on the predictions type (from Selector or from Predictor)
    score_statistics = _get_prediction_score_average_instance(scores[0])
    score_statistics.selector_name = selector_name
    score_statistics.selection_size = sample_score.selection_size

    # Create one score per label
    for label in range(0, n_labels):
        score_by_label = PredictionScoreAverageByLabel()
        score_statistics.by_label.append(score_by_label)

    # Calculate the sum of all scores, general and per label
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    support_list = []
    prediction_by_label = {}
    for label in range(0, n_labels):
        prediction_by_label[label] = {
            'precision_list': [],
            'recall_list': [],
            'f1_list': [],
            'support_list': []
        }

    for score in scores:
        general_score = score.report.general
        accuracy_list.append(general_score.accuracy)
        precision_list.append(general_score.precision)
        recall_list.append(general_score.recall)
        f1_list.append(general_score.f1_score)
        support_list.append(general_score.support)
        for label in range(0, n_labels):
            label_score = score.report.per_label[label]
            label_prediction = prediction_by_label[label]
            label_prediction['precision_list'].append(label_score.precision)
            label_prediction['recall_list'].append(label_score.recall)
            label_prediction['f1_list'].append(label_score.f1_score)
            label_prediction['support_list'].append(label_score.support)

    # Calculate all statistics
    score_statistics.accuracy_avg = np.mean(accuracy_list)
    score_statistics.accuracy_var = np.var(accuracy_list)
    score_statistics.precision_avg = np.mean(precision_list)
    score_statistics.precision_var = np.var(precision_list)
    score_statistics.recall_avg = np.mean(recall_list)
    score_statistics.recall_var = np.var(recall_list)
    score_statistics.f1_score_avg = np.mean(f1_list)
    score_statistics.f1_score_var = np.var(f1_list)
    score_statistics.support_avg = np.mean(support_list)
    score_statistics.support_var = np.var(support_list)
    for label in range(0, n_labels):
        label_prediction = prediction_by_label[label]
        score_by_label = score_statistics.by_label[label]
        score_by_label.label = label
        score_by_label.precision_avg = np.mean(label_prediction['precision_list'])
        score_by_label.precision_var = np.var(label_prediction['precision_list'])
        score_by_label.recall_avg = np.mean(label_prediction['recall_list'])
        score_by_label.recall_var = np.var(label_prediction['recall_list'])
        score_by_label.f1_score_avg = np.mean(label_prediction['f1_list'])
        score_by_label.f1_score_var = np.var(label_prediction['f1_list'])
        score_by_label.support_avg = np.mean(label_prediction['support_list'])
        score_by_label.support_var = np.var(label_prediction['support_list'])

    if isinstance(score_statistics, SelectorPredictionScoreStatistics):
        print_with_time(f'Selector/Predictor average general f1 score: {score_statistics.f1_score_avg} +/- {score_statistics.f1_score_var}')

    return score_statistics

def _get_prediction_score_average_instance(sample_score: BasePredictionScore):
    '''
    Given a score, defines the correct average format (by selector or by predictor)
    '''
    if isinstance(sample_score, SelectorPredictionScore):
        return SelectorPredictionScoreStatistics()
    elif isinstance(sample_score, PredictorPredictionScore):
        return PredictorPredictionScoreStatistics()
    else:
        raise ValueError("Invalid prediction score")