import uuid
import hydra
import matplotlib as plt
from config.config import Config
import config.general_config as general_config
from typing import List
from evaluation.occlusion.OcclusionScore import OcclusionScore, OcclusionScorePerLabel
from util.output_folder import create_output_files
from util.selection_persistence import persist_rank, persist_weights, persist_execution_metrics
from util.performance_util import ExecutionTimeCounter
from util.print_util import print_with_time
from util.dict_util import add_on_dict_list
from data.DataLoader import load_dataset
from data.DataSplitter import get_train_and_test_data_from_dataframe, get_dataset_with_k_fold
from history.ExecutionHistory import ExecutionHistory
from evaluation.execution_time.ExecutionTime import create_execution_time_table_and_chart
from evaluation.informative_features.InformativeFeaturesScore import InformativeFeaturesScore
from evaluation.informative_features.InformativeFeatures import calculate_informative_features_scores, create_heatmap, create_informative_features_scores_output
from evaluation.stability.Stability import calculate_stability_scores, create_stability_table_and_charts, generate_feature_selection_stability_chart
from evaluation.stability.StabilityScore import StabilityScore
from evaluation.prediction.Prediction import calculate_prediction_score_from_selector, calculate_prediction_average_from_selector, calculate_prediction_scores_from_feature_selection, create_selectors_prediction_average_table_and_chart, create_predictors_table_and_chart, create_selectors_prediction_chart
from evaluation.prediction.PredictionAverage import SelectorPredictionScoreStatistics, PredictorPredictionScoreStatistics
from evaluation.prediction.PredictionScore import SelectorPredictionScore
from evaluation.occlusion.Occlusion import calculate_and_persist_occlusion, persist_merged_occlusion
from selector.enum.PredictionMode import PredictionMode


# Feature Selectors
from selector.LassoSelector import LassoSelectorWrapper #Rápido
from selector.DecisionTreeSelector import DecisionTreeSelectorWrapper #Rápido
from selector.RandomForestSelector import RandomForestSelectorWrapper #Rápido
from selector.SHAPSelector import SHAPSelectorWrapper #Médio
from selector.LIMESelector import LIMESelectorWrapper #Lento+
from selector.DeepSHAPSelector import DeepSHAPSelectorWrapper #Médio
from selector.KruskalWallisSelector import KruskalWallisSelectorWrapper
from selector.ReliefFSelector import ReliefFSelectorWrapper
from selector.FeatureSelectionLayerSelector import FeatureSelectionLayerSelectorWrapper
from selector.fs_based.FeatureSelectionObserverSelector import FeatureSelectionObserverSelectorWrapper
from selector.fs_based._export import RFSLayerSelectorV1Wrapper
from selector.fs_based._export import MFSLayerV1ReLUSelector
from selector.fs_based._export import MFSLayerV1SigmoidSelector
from selector.fs_based._export import MFSLayerV1TanhSelector
from selector.fs_based._export import FSRLayerV1ReLUSelector
from selector.fs_based._export import FSRLayerV1SigmoidSelector
from selector.fs_based._export import FSRLayerV1TanhSelector


@hydra.main(version_base=None, config_path="config", config_name="config")
def execute_experiment(config: Config):
    # Create execution id
    execution_id = str(uuid.uuid4())
    print(f'Execution ID: {execution_id}')

    # Create folder to save results
    create_output_files(config.output, execution_id)

    return

    # Disable Matplot open figures alert as more than 20 figures are necessary to generate video
    plt.rcParams.update({'figure.max_open_warning': 0})

    # Start total execution time
    execution_time_counter = ExecutionTimeCounter().start()

    # Load data
    dataframe = load_dataset(general_config.DATASET_FILE)

    # Define dataset
    splitted_dataset = get_train_and_test_data_from_dataframe(dataframe, test_size=general_config.TEST_SIZE)
    general_train_dataset = splitted_dataset.get_train()
    test_dataset = splitted_dataset.get_test()

    # Define KFold datasets
    train_datasets = get_dataset_with_k_fold(general_train_dataset, general_config.K_FOLD, general_config.K_FOLD_REPEAT)

    # Define selector
    #selectors_types = [MFSLayerV1ReLUSelector, FeatureSelectionLayerSelectorWrapper, FeatureSelectionObserverSelectorWrapper, RFSLayerSelectorV1Wrapper, FSRLayerV1ReLUSelector]
    #selectors_types = [MFSLayerV1ReLUSelector, LassoSelectorWrapper, DecisionTreeSelectorWrapper, RandomForestSelectorWrapper, LIMESelectorWrapper, DeepSHAPSelectorWrapper, KruskalWallisSelectorWrapper, ReliefFSelectorWrapper]
    selectors_types = [MFSLayerV1ReLUSelector, LassoSelectorWrapper, LIMESelectorWrapper, DeepSHAPSelectorWrapper]

    # Validate unique names between selectors
    names = [selector_type.get_name() for selector_type in selectors_types]
    if len(names) > len(set(names)):
        raise ValueError("Selectors name aren't unique")

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
            selector.fit(train_dataset, test_dataset)
        
            # Calculate and store execution time
            selector_execution_time = selector_execution_time_counter.print_end('Selector training').get_execution_time()
            add_on_dict_list(execution_times_by_selector, selector_type.get_name(), selector_execution_time)

            print_with_time(f'Calculating selector round metrics...')

            # Calculate prediction score of selector if available
            prediction_score = None
            if selector.get_prediction_mode() == PredictionMode.AVAILABLE:
                prediction_score = calculate_prediction_score_from_selector(selector, test_dataset)
                add_on_dict_list(selector_prediction_scores_by_selector, selector.get_class_name(), prediction_score)
                f1_score = prediction_score.report.general.f1_score
                if f1_score > best_score:
                    best_selector = selector
                    best_score = f1_score
                    best_score_index = i
            else:
                best_selector = selector
                best_score_index = i

            # Add selection results on history
            history.add(selector, splitted_dataset, selector_execution_time, prediction_score)
        
        if best_selector.get_prediction_mode() == PredictionMode.AVAILABLE:
            print_with_time(f'Best F1 score: {best_score}')
        
        print_with_time(f'Calculating selector aggregated metrics...')

        # Persist heatmap
        create_heatmap(best_selector, train_dataset)

        # Calculate selector prediction average
        if best_selector.get_prediction_mode() == PredictionMode.AVAILABLE:
            selector_prediction_score_average = calculate_prediction_average_from_selector(history)
            selector_prediction_score_average_by_selector[best_selector.get_class_name()] = selector_prediction_score_average

        # Persist weights
        persist_weights(history, test_dataset.get_feature_names(), best_score_index)

        # Persist ranks
        persist_rank(history, test_dataset.get_feature_names(), best_score_index)

        # Calculate occlusion
        if best_selector.get_prediction_mode() == PredictionMode.AVAILABLE:
            selector_occlusion_scores, selector_occlusion_scores_per_class = calculate_and_persist_occlusion(best_selector, test_dataset)
            occlusion_scores.extend(selector_occlusion_scores)
            occlusion_by_label_scores.extend(selector_occlusion_scores_per_class)

        # Calculate selected informative features percentages
        informative_features_scores = calculate_informative_features_scores(history, test_dataset)
        informative_scores_by_selector[best_selector.get_class_name()] = informative_features_scores

        # Calculate prediction average using different predictors
        predictors_scores = calculate_prediction_scores_from_feature_selection(best_selector, splitted_dataset)
        predictors_scores_by_selector[best_selector.get_class_name()] = predictors_scores

        # Calculate stability
        stability_scores = calculate_stability_scores(history)
        stability_scores_by_selector[best_selector.get_class_name()] = stability_scores

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