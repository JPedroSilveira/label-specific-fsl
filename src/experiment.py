import selectors
import uuid
import hydra
import matplotlib as plt
from skimage import data

from config.type import Config
from src.domain.selector.SelectorTypeCreator import SelectorTypeCreator
from src.domain.data.DatasetsCreator import DatasetsCreator
from src.domain.data.KFoldCreator import KFoldCreator
from src.domain.data.DatasetLoader import DatasetLoader
from src.domain.folder.OutputFolderCreator import OutputFolderCreator

# Non refactored imports

from typing import List
from src.evaluation.occlusion.OcclusionScore import OcclusionScore, OcclusionScorePerLabel
from src.util.selection_persistence import persist_rank, persist_weights, persist_execution_metrics
from src.util.performance_util import ExecutionTimeCounter
from src.util.print_util import print_with_time
from src.util.dict_util import add_on_dict_list
from src.history.ExecutionHistory import ExecutionHistory
from src.evaluation.execution_time.ExecutionTime import create_execution_time_table_and_chart
from src.evaluation.informative_features.InformativeFeaturesScore import InformativeFeaturesScore
from src.evaluation.informative_features.InformativeFeatures import calculate_informative_features_scores, create_heatmap, create_informative_features_scores_output
from src.evaluation.stability.Stability import calculate_stability_scores, create_stability_table_and_charts, generate_feature_selection_stability_chart
from src.evaluation.stability.StabilityScore import StabilityScore
from src.evaluation.prediction.Prediction import calculate_prediction_score_from_selector, calculate_prediction_average_from_selector, calculate_prediction_scores_from_feature_selection, create_selectors_prediction_average_table_and_chart, create_predictors_table_and_chart, create_selectors_prediction_chart
from src.evaluation.prediction.PredictionAverage import SelectorPredictionScoreStatistics, PredictorPredictionScoreStatistics
from src.evaluation.prediction.PredictionScore import SelectorPredictionScore
from src.evaluation.occlusion.Occlusion import calculate_and_persist_occlusion, persist_merged_occlusion


@hydra.main(version_base=None, config_path="config", config_name="config")
def execute_experiment(config: Config) -> None:
    # Disable Matplot open figures alert as more than 20 figures are necessary to generate video
    plt.rcParams.update({'figure.max_open_warning': 0})
    
    # Create execution id
    execution_id = str(uuid.uuid4())
    print(f'Execution ID: {execution_id}')

    # Create outputs folders to save results
    OutputFolderCreator.execute(config.output, execution_id)
    
    # Start total execution time
    execution_time_counter = ExecutionTimeCounter().start()

    # Load data
    dataframe = DatasetLoader.execute(config.dataset)
    
    # Split data into train and test sets
    splitted_dataset = DatasetsCreator.execute(dataframe, config)
    
    print(f"Dataset features: {splitted_dataset.get_n_features()}")
    print(f"Dataset labels: {splitted_dataset.get_n_labels()}")
    print(f"Train samples: {splitted_dataset.get_train().get_n_samples()}")
    print(f"Test samples: {splitted_dataset.get_test().get_n_samples()}")

    # Define KFold datasets
    k_datasets = KFoldCreator.execute(splitted_dataset.get_train(), config)
    print(f'K-Fold datasets created: {len(k_datasets)}')
    
    # Define selectors
    selectors_types = SelectorTypeCreator.execute(config)
    print("Defined selectors:")
    for selectors_type in selectors_types:
        print(f'- {selectors_type.get_name()}')
    
    return

    # Initialize metrics dictionaries per selector
    execution_times_by_selector: dict[str, List[float]] = {}
    selector_prediction_score_average_by_selector: dict[str, SelectorPredictionScoreStatistics] = {}
    selector_prediction_scores_by_selector: dict[str, List[SelectorPredictionScore]] = {}
    predictors_scores_by_selector: dict[str, List[PredictorPredictionScoreStatistics]] = {}
    stability_scores_by_selector: dict[str, List[StabilityScore]] = {}
    informative_scores_by_selector: dict[str, List[InformativeFeaturesScore]] = {}
    occlusion_scores: List[OcclusionScore] = []
    occlusion_by_label_scores: List[List[OcclusionScorePerLabel]] = []

    # Calculate metrics for each selector
    for selector_type in selectors_types:
        print_with_time(f'Running for selector {selector_type.get_name()}')

        # Create history
        history = ExecutionHistory()

        # Persist the best selector based on F1 score if available, else uses the last one
        best_selector = None
        best_score = -1
        best_score_index = -1
        print(f'Total rounds: {len(train_datasets)}')
        print_with_time('Training...')
        for i, train_dataset in enumerate(train_datasets):
            print_with_time(f'Round {i}')

            # Create an instance of selector
            selector = selector_type(train_dataset.get_n_features(), train_dataset.get_n_labels())

            # Start execution timer
            selector_execution_time_counter = ExecutionTimeCounter().start()

            # Train selector or base model
            src.selector.fit(train_dataset, test_dataset)
        
            # Calculate and store execution time
            selector_execution_time = selector_execution_time_counter.print_end('Selector training').get_execution_time()
            add_on_dict_list(execution_times_by_selector, selector_type.get_name(), selector_execution_time)

            print_with_time(f'Calculating selector round metrics...')

            # Calculate prediction score of selector if available
            prediction_score = None
            if src.selector.get_prediction_mode() == PredictionMode.AVAILABLE:
                prediction_score = calculate_prediction_score_from_selector(selector, test_dataset)
                add_on_dict_list(selector_prediction_scores_by_selector, src.selector.get_class_name(), prediction_score)
                f1_score = prediction_score.report.general.f1_score
                if f1_score > best_score:
                    best_selector = selector
                    best_score = f1_score
                    best_score_index = i
            else:
                best_selector = selector
                best_score_index = i

            # Add selection results on history
            src.history.add(selector, splitted_dataset, selector_execution_time, prediction_score)
        
        if best_src.selector.get_prediction_mode() == PredictionMode.AVAILABLE:
            print_with_time(f'Best F1 score: {best_score}')
        
        print_with_time(f'Calculating selector aggregated metrics...')

        # Persist heatmap
        create_heatmap(best_selector, train_dataset)

        # Calculate selector prediction average
        if best_src.selector.get_prediction_mode() == PredictionMode.AVAILABLE:
            selector_prediction_score_average = calculate_prediction_average_from_selector(history)
            selector_prediction_score_average_by_selector[best_src.selector.get_class_name()] = selector_prediction_score_average

        # Persist weights
        persist_weights(history, test_dataset.get_feature_names(), best_score_index)

        # Persist ranks
        persist_rank(history, test_dataset.get_feature_names(), best_score_index)

        # Calculate occlusion
        if best_src.selector.get_prediction_mode() == PredictionMode.AVAILABLE:
            selector_occlusion_scores, selector_occlusion_scores_per_class = calculate_and_persist_occlusion(best_selector, test_dataset)
            occlusion_scores.extend(selector_occlusion_scores)
            occlusion_by_label_scores.extend(selector_occlusion_scores_per_class)

        # Calculate selected informative features percentages
        informative_features_scores = calculate_informative_features_scores(history, test_dataset)
        informative_scores_by_selector[best_src.selector.get_class_name()] = informative_features_scores

        # Calculate prediction average using different predictors
        predictors_scores = calculate_prediction_scores_from_feature_selection(best_selector, splitted_dataset)
        predictors_scores_by_selector[best_src.selector.get_class_name()] = predictors_scores

        # Calculate stability
        stability_scores = calculate_stability_scores(history)
        stability_scores_by_selector[best_src.selector.get_class_name()] = stability_scores

        # Create stability metrics
        print_with_time(f'Creating feature selections output...')
        generate_feature_selection_stability_chart(best_selector, history, test_dataset.get_feature_names())

    # Persist raw metrics
    print_with_time(f'Persisting raw metrics...')
    persist_execution_metrics(execution_times_by_selector, selector_prediction_scores_by_selector)
    
    # Create comparison analysis
    print_with_time(f'Calculating selectors comparative metrics...')
    print_with_time(f'Creating merged oclusion output...')
    persist_merged_occlusion(occlusion_scores, occlusion_by_label_scores)
    print_with_time(f'Creating execution time output...')
    create_execution_time_table_and_chart(execution_times_by_selector)
    print_with_time(f'Creating informative features output...')
    create_informative_features_scores_output(informative_scores_by_selector)
    print_with_time(f'Creating prediction scores output...')
    create_selectors_prediction_chart(selector_prediction_scores_by_selector)
    create_selectors_prediction_average_table_and_chart(selector_prediction_score_average_by_selector)
    create_predictors_table_and_chart(predictors_scores_by_selector, splitted_dataset.get_n_features())
    print_with_time(f'Creating stability scores output...')
    create_stability_table_and_charts(stability_scores_by_selector, splitted_dataset.get_n_features())
    
    # Display total execution time
    execution_time_counter.print_end('Global')