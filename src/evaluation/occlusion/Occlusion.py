from typing import List
import torch
from torch import nn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from src.util.device_util import get_device
from src.util.dict_util import add_on_dict_list
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
from src.config.general_config import OCCLUSION_INITIAL_END, OCCLUSION_STEP, SHOULD_CALCULATE_METRICS_BY_LABEL, OCCLUSION_BY_LOSS, OCCLUSION_LIMIT, OCCLUSION_INITIAL_STEP, OUTPUT_PATH, OCCLUSION_OUTPUT_SUB_PATH, SHOULD_SAVE_OCCLUSION_VIDEO, OCCLUSION_STEP_ON_CHART
from src.data.Dataset import Dataset
from src.evaluation.occlusion.OcclusionScore import OcclusionScore, OcclusionScorePerLabel
from src.selector.enum.SelectionSpecificity import SelectionSpecificity
from src.util.feature_selection_util import remove_n_features_from_inversed_rank, remove_n_features_from_rank
from src.selector.BaseSelectorWrapper import BaseSelectorWrapper
from src.util.classification_report_util import calculate_classification_report
from src.util.print_util import print_load_bar
from src.util.plot_util import save_plots_as_video
from src.util.performance_util import ExecutionTimeCounter
from src.util.numpy_util import convert_nparray_to_tensor


def calculate_and_persist_occlusion(selector: BaseSelectorWrapper, dataset: Dataset):
    '''
    Given a selector and a dataset, calculate prediction metrics when ocluding important features based on selectors ranking
    '''
    scores: list[OcclusionScore] = []
    scores_per_class: list[list[OcclusionScorePerLabel]] = []
    # Calculate occlusion for general ranking
    if SelectionSpecificity.GENERAL in src.selector.get_selection_specificities():
        time_counter = ExecutionTimeCounter().print_start('Calculating general occlusion metrics...')
        scores = _calculate_occlusion_general(selector, dataset)
        time_counter.print_end('General occlusion test')
    # Calculate occlusion for label ranking
    if SHOULD_CALCULATE_METRICS_BY_LABEL and SelectionSpecificity.PER_LABEL in src.selector.get_selection_specificities():
        time_counter = ExecutionTimeCounter().print_start('Calculating per label occlusion metrics...')
        scores_per_class = _calculate_occlusion_per_label(selector, dataset)
        time_counter.print_end('Per label occlusion test')
    # Persist results as a chart
    _persist_occlusion_metrics_as_a_chart(selector, scores, scores_per_class)
    return scores, scores_per_class

def persist_merged_occlusion(occlusion_scores: List[OcclusionScore], occlusion_by_label_scores: List[List[OcclusionScorePerLabel]]):
    _persist_merged_general_occlusion_as_a_chart(occlusion_scores)
    _persist_merged_per_label_occlusion_as_a_chart(occlusion_by_label_scores)

def _persist_merged_per_label_occlusion_as_a_chart(occlusion_by_label_scores: List[List[OcclusionScorePerLabel]]):
    '''
    Persist the occlusion metrics for each label in a isolated chart
    '''
    scores_by_label: dict[str, List[OcclusionScorePerLabel]] = {}
    for scores in occlusion_by_label_scores:
        for score in scores:
            add_on_dict_list(scores_by_label, score.label, score)

    for label in scores_by_label.keys():
        scores = scores_by_label[label]
        data = {
            'Algorithm': [],
            'Removed Features': [],
            'Score': []
        }
        for score in scores:
            for label_score in score.report.per_label:
                data['Algorithm'].append(f'{score.selector_name} - Label {label_score.label}')
                data['Removed Features'].append(score.removed_features)
                data['Score'].append(label_score.f1_score)
        _persist_occlusion_chart(data, f'merged-label-{label}-occlusion')
        if OCCLUSION_BY_LOSS:
            data = {
                'Algorithm': [],
                'Removed Features': [],
                'Score': []
            }
            for score in scores:
                for label_score in score.report.per_label:
                    data['Algorithm'].append(f'{score.selector_name} - Label {label_score.label}')
                    data['Removed Features'].append(score.removed_features)
                    data['Score'].append(score.loss_by_label[label_score.label])
            _persist_occlusion_chart(data, f'merged-label-{label}-occlusion-by-loss', isLoss=True)
        data = {
            'Algorithm': [],
            'Removed Features': [],
            'Score': []
        }
        for score in scores:
            for label_score in score.inverse_report.per_label:
                data['Algorithm'].append(f'{score.selector_name} - Label {label_score.label}')
                data['Removed Features'].append(score.removed_features)
                data['Score'].append(label_score.f1_score)
        _persist_occlusion_chart(data, f'merged-label-{label}-inversed-occlusion')

def _persist_merged_general_occlusion_as_a_chart(occlusion_scores: List[OcclusionScore]):
    '''
    Persist the occlusion metrics for general score, including all labels
    '''
    data = {
        'Algorithm': [],
        'Removed Features': [],
        'Score': []
    }
    for score in occlusion_scores:
        data['Algorithm'].append(f'{score.selector_name}')
        data['Removed Features'].append(score.removed_features)
        data['Score'].append(score.report.general.f1_score)
    _persist_occlusion_chart(data, f'Merged - Label General - Occlusion')
    if OCCLUSION_BY_LOSS:
        data = {
            'Algorithm': [],
            'Removed Features': [],
            'Score': []
        }
        for score in occlusion_scores:
            data['Algorithm'].append(f'{score.selector_name}')
            data['Removed Features'].append(score.removed_features)
            data['Score'].append(score.loss)
        _persist_occlusion_chart(data, f'Merged - Label General - Occlusion - By Loss', isLoss=True)
    data = {
        'Algorithm': [],
        'Removed Features': [],
        'Score': []
    }
    for score in occlusion_scores:
        data['Algorithm'].append(f'{score.selector_name}')
        data['Removed Features'].append(score.removed_features)
        data['Score'].append(score.inverse_report.general.f1_score) 
    _persist_occlusion_chart(data, f'Merged - Label General - Occlusion Inversed')

def _get_dataset_with_occlusions(dataset: Dataset, features_to_include: list[int]):
    '''
    Given a dataset and a list of features to oclude, returns a new dataset with the removed feature values set to zero
    '''
    mask = np.zeros_like(dataset.get_features())
    mask[:, features_to_include] = 1
    features = dataset.get_features() * mask
    return Dataset(
        features=features, 
        labels=dataset.get_labels(), 
        label_types=dataset.get_label_types(), 
        feature_names=dataset.get_feature_names(),
        informative_features=dataset.get_informative_features(), 
        informative_features_per_label=dataset.get_informative_features_per_label()
    )

def _calculate_loss_by_label(dataset: Dataset, losses: np.ndarray):
    """
    Calculates the loss per label.

    Args:
        y_true: True labels as a one-hot encoded numpy array.
        y_pred: Predicted probabilities as a numpy array.

    Returns:
        A dictionary mapping labels to their corresponding average losses.
    """
    loss_per_label: dict[int, float] = {}
    # For each unique label
    for label in range(0, dataset.get_n_labels()):
        # Get all indexes from the dataset related to the label
        indexes_for_label = []
        for i, value in enumerate(dataset.get_labels()):
            if value == label:
                indexes_for_label.append(i)
        # Calculate loss when labels are available
        if len(indexes_for_label) == 0:
            print(f'Warning! Fail to calculate loss for label {label}, zero examples where found on dataset')
            loss_per_label[label] = 0.0
        else:
            # Calculate the loss for the items related to the label using mean
            label_losses = losses[indexes_for_label]
            loss_per_label[label] = np.mean(label_losses)
    return loss_per_label

def _calculate_score_removing_features_based_on_ranking(selector: BaseSelectorWrapper, dataset: Dataset, rank: np.ndarray, n_ocluded_features: int):
    '''
    Given a rank and a number of features to remove, calculate the prediction score when ocluding the features beased on ranking order
    '''
    # Rank order
    features_to_include = remove_n_features_from_rank(rank, n_ocluded_features)
    filtered_dataset = _get_dataset_with_occlusions(dataset, features_to_include)
    y_pred_probabilities = src.selector.predict_probabilities(filtered_dataset, use_softmax=False)
    y_pred = np.argmax(y_pred_probabilities, 1)
    classification_report = calculate_classification_report(dataset, y_pred)
    # Create a figure with the confusion matrix
    if SHOULD_SAVE_OCCLUSION_VIDEO:
        figure = _create_confusion_matrix_figure(dataset, y_pred, n_ocluded_features)
    else:
        figure = None
    # Rank inverse order
    features_to_include = remove_n_features_from_inversed_rank(rank, n_ocluded_features)
    filtered_dataset = _get_dataset_with_occlusions(dataset, features_to_include)
    y_pred_probabilities = src.selector.predict_probabilities(filtered_dataset, use_softmax=False)
    y_pred = np.argmax(y_pred_probabilities, 1)
    inversed_classification_report = calculate_classification_report(dataset, y_pred)
    if OCCLUSION_BY_LOSS:
        # Calculate losses for each prediction
        device = get_device()
        label_weights=compute_class_weight(class_weight="balanced", classes=dataset.get_label_types(), y=dataset.get_labels())
        label_weights=torch.tensor(label_weights, dtype=torch.float).to(device)
        cross_entropy_loss = nn.CrossEntropyLoss(reduction='none', weight=label_weights)
        losses = cross_entropy_loss(convert_nparray_to_tensor(y_pred_probabilities, data_type=torch.float32), convert_nparray_to_tensor(dataset.get_labels(), data_type=torch.long)).cpu().numpy()
        # Calculate general loss using mean
        general_loss = np.mean(losses)
        # Calculate by label loss using mean
        loss_by_label = _calculate_loss_by_label(dataset, losses)
    else:
        general_loss = None
        loss_by_label = None
    return classification_report, general_loss, loss_by_label, figure, inversed_classification_report

def _create_confusion_matrix_figure(dataset: Dataset, y_pred: np.ndarray, n_ocluded_features: int):
    '''
    Create a confusion matrix based on a set of predictions
    '''
    confusion = confusion_matrix(dataset.get_labels(), y_pred)
    figure = ConfusionMatrixDisplay(confusion).plot().figure_
    figure.suptitle(f'Ocluded labels: {n_ocluded_features}')
    return figure

def _calculate_general_occlusion_score_and_confusion(selector: BaseSelectorWrapper, dataset: Dataset, rank: np.ndarray, n_ocluded_features: int):
    report, loss, loss_by_label, confusion_matrix, inversed_report = _calculate_score_removing_features_based_on_ranking(selector, dataset, rank, n_ocluded_features)
    score = OcclusionScore(selector, n_ocluded_features, report, loss, loss_by_label, inversed_report)
    return score, confusion_matrix

def _get_occlusion_steps(dataset: Dataset):
    limit = OCCLUSION_LIMIT if OCCLUSION_LIMIT != None else dataset.get_n_features()
    steps = list(range(0, OCCLUSION_INITIAL_END, OCCLUSION_INITIAL_STEP))
    last_steps = list(range(OCCLUSION_INITIAL_END, limit, OCCLUSION_STEP))
    steps.extend(last_steps)
    if steps[-1] != limit:
        steps.append(limit)
    return steps

def _calculate_occlusion_general(selector: BaseSelectorWrapper, dataset: Dataset):
    '''
    Calculate the occlusion based on general ranking
    '''
    scores: list[OcclusionScore] = []
    confusion_matrix_list: list[Figure] = []
    # Calculates the prediction score for each amount of ocluded features
    occlusion_steps = _get_occlusion_steps(dataset)
    rank = src.selector.get_general_ranking()
    for i, n_ocluded_features in enumerate(occlusion_steps):
        print_load_bar(i + 1, len(occlusion_steps))
        score, confusion_matrix = _calculate_general_occlusion_score_and_confusion(selector, dataset, rank, n_ocluded_features)
        scores.append(score)
        confusion_matrix_list.append(confusion_matrix)
    # Persist the confusions matrix list as a video
    if SHOULD_SAVE_OCCLUSION_VIDEO:
        save_plots_as_video(
            file_name=f'{OUTPUT_PATH}/{OCCLUSION_OUTPUT_SUB_PATH}/{src.selector.get_class_name()}-general-confusion-matrix.gif',
            plot_list=confusion_matrix_list
        )
    return scores

def _calculate_per_label_occlusion_score_and_confusion(selector: BaseSelectorWrapper, dataset: Dataset, rank: np.ndarray, label: int, n_ocluded_features: int):
    report, loss, loss_by_label, confusion_matrix, inversed_report = _calculate_score_removing_features_based_on_ranking(selector, dataset, rank, n_ocluded_features)
    score = OcclusionScorePerLabel(selector, n_ocluded_features, label, report, loss, loss_by_label, inversed_report)
    return score, confusion_matrix

def _calculate_occlusion_per_label(selector: BaseSelectorWrapper, dataset: Dataset):
    scores_per_label: list[list[OcclusionScorePerLabel]] = []
    rank_per_class = src.selector.get_ranking_per_class()
    # Calculate occlusion based on selector rank for each label
    for label, rank in enumerate(rank_per_class):
        print(f"Calculating occlusion for label {label}")
        scores = []
        confusion_matrix_list = []
        # Calculates the prediction score for each amount of ocluded features
        occlusion_steps = _get_occlusion_steps(dataset)
        for i, n_ocluded_features in enumerate(occlusion_steps):
            print_load_bar(i + 1, len(occlusion_steps))
            score, confusion_matrix = _calculate_per_label_occlusion_score_and_confusion(selector, dataset, rank, label, n_ocluded_features)
            scores.append(score)
            confusion_matrix_list.append(confusion_matrix)  
        scores_per_label.append(scores)
        # Persist the confusions matrix list as a video
        if SHOULD_SAVE_OCCLUSION_VIDEO:
            save_plots_as_video(
                file_name=f'{OUTPUT_PATH}/{OCCLUSION_OUTPUT_SUB_PATH}/{src.selector.get_class_name()}-per-label-{label}-confusion-matrix.gif',
                plot_list=confusion_matrix_list
            )
    return scores_per_label

def _persist_occlusion_chart(data: dict[str, list], title: str, column='Algorithm', isLoss: bool=False):
    '''
    Given a data dictionary with the occlusion metrics for each label (or general), create and persist a chart
    '''
    max_number_of_removed_features = max(data['Removed Features'])
    df = pd.DataFrame(data)
    df_pivot = df.pivot(index="Removed Features", columns=column, values="Score")
    # Create lineplot
    sns.lineplot(data=df_pivot)
    # Set figure title legends
    plt.title('')
    plt.xlabel("Number of erased features")
    plt.ylabel("Loss (test)" if isLoss else "F1 score (test)")
    # Show values increasing 0.1 by 0.1 on Y axis
    if not isLoss:
        plt.yticks(np.arange(0.0, 1.05, 0.1))
    # Set legend place outside the graph
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # Enable grid
    plt.grid(True)
    # Show values increasing one by one on X axis
    plt.xticks(np.arange(0, max_number_of_removed_features + 1, OCCLUSION_STEP_ON_CHART))
    # Persist chart as image
    plt.savefig(f'{OUTPUT_PATH}/{OCCLUSION_OUTPUT_SUB_PATH}/{title}.pdf', format='pdf', bbox_inches='tight')
    # Reset plot
    plt.close()

def _persist_general_occlusion_metrics_as_a_chart(selector: BaseSelectorWrapper, scores: list[OcclusionScore]):
    '''
    Persist the occlusion metrics for general score, including all labels
    '''
    data = {
        'Label': [],
        'Removed Features': [],
        'Score': []
    }
    for score in scores:
        data['Label'].append('General')
        data['Removed Features'].append(score.removed_features)
        data['Score'].append(score.report.general.f1_score)
    _persist_occlusion_chart(data, f'{src.selector.get_class_name()} - Label General - Occlusion', column='Label')
    data = {
        'Label': [],
        'Removed Features': [],
        'Score': []
    }
    for score in scores:
        data['Label'].append('General')
        data['Removed Features'].append(score.removed_features)
        data['Score'].append(score.loss)
    _persist_occlusion_chart(data, f'{src.selector.get_class_name()} - Label General - Occlusion - By Loss', column='Label', isLoss=True)
    data = {
        'Label': [],
        'Removed Features': [],
        'Score': []
    }
    for score in scores:
        data['Label'].append('General')
        data['Removed Features'].append(score.removed_features)
        data['Score'].append(score.inverse_report.general.f1_score) 
    _persist_occlusion_chart(data, f'{src.selector.get_class_name()} - Label General - Occlusion - Inversed', column='Label')

def _persist_per_label_occlusion_metrics_as_a_chart(selector: BaseSelectorWrapper, scores_per_label: list[list[OcclusionScorePerLabel]]):
    '''
    Persist the occlusion metrics for each label in a isolated chart
    '''
    for label, scores in enumerate(scores_per_label):
        data = {
            'Label': [],
            'Removed Features': [],
            'Score': []
        }
        for score in scores:
            for label_score in score.report.per_label:
                data['Label'].append(f'Label {label_score.label}')
                data['Removed Features'].append(score.removed_features)
                data['Score'].append(label_score.f1_score)
        _persist_occlusion_chart(data, f'{src.selector.get_class_name()}-label-{label}-occlusion', column='Label')
        if OCCLUSION_BY_LOSS:
            data = {
                'Label': [],
                'Removed Features': [],
                'Score': []
            }
            for score in scores:
                for label_score in score.report.per_label:
                    data['Label'].append(f'Label {label_score.label}')
                    data['Removed Features'].append(score.removed_features)
                    data['Score'].append(score.loss_by_label[label_score.label])
            _persist_occlusion_chart(data, f'{src.selector.get_class_name()}-label-{label}-occlusion-by-loss', column='Label', isLoss=True)
        data = {
            'Label': [],
            'Removed Features': [],
            'Score': []
        }
        for score in scores:
            for label_score in score.inverse_report.per_label:
                data['Label'].append(f'Label {label_score.label}')
                data['Removed Features'].append(score.removed_features)
                data['Score'].append(label_score.f1_score)
        _persist_occlusion_chart(data, f'{src.selector.get_class_name()}-label-{label}-occlusion-inversed', column='Label')
        

def _persist_occlusion_metrics_as_a_chart(selector: BaseSelectorWrapper, scores: list[OcclusionScore], scores_per_class: list[list[OcclusionScorePerLabel]]):
    '''
    Persist occlusion metrics as a chart
    '''
    if SelectionSpecificity.GENERAL in src.selector.get_selection_specificities():
        _persist_general_occlusion_metrics_as_a_chart(selector, scores)
    if SelectionSpecificity.PER_LABEL in src.selector.get_selection_specificities():
        _persist_per_label_occlusion_metrics_as_a_chart(selector, scores_per_class)

