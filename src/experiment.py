import selectors
import uuid
import hydra
import matplotlib as plt
from typing import List

from config.type import Config
from src.domain.wtsne.WTSNECreator import WTSNECreator
from src.domain.data.LabelConverter import LabelConverter
from src.domain.stability.ReducedSetStabilityResultAggregator import ReducedSetStabilityResultAggregator
from src.domain.stability.ReducedSetStabilityMetric import ReducedSetStabilityMetric
from src.domain.timer.ExecutionTimeStats import ExecutionTimeStats
from src.domain.storage.ExecutionStorage import ExecutionStorage
from src.domain.pytorch.TestGPU import TestGPU
from src.domain.log.Logger import Logger
from src.domain.stability.SelectorStabilityMetric import SelectorStabilityMetric
from src.domain.prediction.SelectorPredictionMetric import SelectorPredictionMetric
from src.domain.seed.SeedSetter import SeedSetter
from src.domain.data.DatasetNormalizer import DatasetNormalizer
from src.domain.data.DatasetScaler import DatasetScaler
from src.domain.weight.WeightPersistence import WeightPersistence
from src.domain.selector.SelectorTypeCreator import SelectorTypeCreator
from src.domain.data.DatasetsCreator import DatasetsCreator
from src.domain.data.KFoldCreator import KFoldCreator
from src.domain.data.DatasetLoader import DatasetLoader
from src.domain.folder.OutputFolderCreator import OutputFolderCreator
from src.domain.timer.ExecutionTimeCounter import ExecutionTimeCounter
# Non refactored imports


@hydra.main(version_base=None, config_path="config", config_name="config")
def execute_experiment(config: Config) -> None:
    # Set seeds
    SeedSetter.execute(config.dataset)
    
    # Disable Matplot open figures alert as more than 20 figures are necessary to generate video
    plt.rcParams.update({'figure.max_open_warning': 0})
    
    # Create outputs folders to save results
    execution_id = str(uuid.uuid4())
    OutputFolderCreator.execute(config.output, execution_id)
    
    # Start logger
    Logger.setup(config)
    Logger.execute(f'[STARTED] Execution ID: {execution_id}')
    
    # Test GPU
    TestGPU.execute()
    
    # Start total execution time
    execution_time_counter = ExecutionTimeCounter().start()

    # Load data
    dataframe = DatasetLoader.execute(config.dataset)
    
    # Convert labels to long
    LabelConverter.execute(dataframe, config.dataset)
    
    # Split data into train and test sets
    splitted_dataset = DatasetsCreator.execute(dataframe, config)
    Logger.execute(f"Dataset features: {splitted_dataset.get_n_features()}")
    Logger.execute(f"Dataset labels: {splitted_dataset.get_n_labels()}")
    Logger.execute(f"Train samples: {splitted_dataset.get_train().get_n_samples()}")
    Logger.execute(f"Test samples: {splitted_dataset.get_test().get_n_samples()}")

    # Apply normalization
    DatasetNormalizer.execute(splitted_dataset, config.dataset)
    
    # Apply scaling
    DatasetScaler.execute(splitted_dataset, config.dataset)
    
    # Define KFold datasets
    k_train_datasets = KFoldCreator.execute(splitted_dataset.get_train(), config.dataset)
    Logger.execute(f'K-Fold datasets created: {len(k_train_datasets)}')
    
    # Define selectors
    selectors_class = SelectorTypeCreator.execute(config)
    Logger.execute("Defined selectors:")
    for selector_class in selectors_class:
        Logger.execute(f'- {selector_class.get_name()}')
        
    # Get feature names
    feature_names = splitted_dataset.get_test().get_feature_names()
    
    # Create a storage
    storage = ExecutionStorage()
        
    # Train for each fold
    for id, train_dataset in enumerate(k_train_datasets):
        Logger.execute(f'Training fold {id+1} of {len(k_train_datasets)}')
        # Train each selector
        for selector_class in selectors_class:
            Logger.execute(f'================================================')
            Logger.execute(f'Running for selector {selector_class.get_name()}')
            # Create an instance of selector
            selector = selector_class(train_dataset.get_n_features(), train_dataset.get_n_labels(), config.dataset)
            # Start execution timer
            selector_execution_time_counter = ExecutionTimeCounter().start()
            # Fit selector
            selector.fit(train_dataset, splitted_dataset.get_test())
            # Calculate execution time
            selector_execution_time = selector_execution_time_counter.print_end('Selector training').get_execution_time()
            storage.add_execution_time(selector, selector_execution_time)
            # Calculate prediction score of selector if available
            if selector.can_predict():
                prediction_score = SelectorPredictionMetric.execute(selector, splitted_dataset.get_test())
                Logger.execute(f'F1 Score: {prediction_score.report.general.f1_score}')
            # Persist weights
            WeightPersistence.execute(id, selector, config.output, feature_names)
            reduced_stability_score_per_size_per_label_per_metric = ReducedSetStabilityMetric.execute(selector, selector_class, train_dataset, splitted_dataset.get_test(), config)
            storage.add_reduced_stability_score(selector, reduced_stability_score_per_size_per_label_per_metric)
            
    # Calculate metrics
    ExecutionTimeStats.execute(selectors_class, storage)
    SelectorStabilityMetric.execute(selectors_class, config)
    ReducedSetStabilityResultAggregator.execute(selectors_class, storage)
    WTSNECreator.execute(selectors_class, splitted_dataset.get_complete(), config)
    
    # TODO: Implement new stability metric
    # TODO: Change @staticmethod to @classmethod
    # TODO: Implement all datasets [XOR, SynthA, SynthB, SynthC, Liver, Colorectal, Breast]
    # TODO: Store prediction performance
    # TODO: Implement WTSNE + silhouette
    # TODO: Implement Feature ranking positions  
    # TODO: Implement PIFS and PSFI metrics with graph
    # TODO: Implement Heatmap
    # TODO: Implement Predictor training with graph (SVC)
    # TODO: Implement Feature erasure
    
    Logger.execute(f'[COMPLETED] Execution ID: {execution_id}')
    Logger.execute(f'- Time: {execution_time_counter.get_execution_time()}s')
            
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