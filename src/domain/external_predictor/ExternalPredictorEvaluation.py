from typing import List, Type

import pandas as pd
from sklearn.svm import SVC

from config.type import DatasetConfig
from src.domain.classification_report.ClassificationReportCalculator import ClassificationReportCalculator
from src.domain.log.Logger import Logger
from src.domain.data.reader.RankingReader import RankingReader
from src.domain.data.types.SplittedDataset import SplittedDataset
from src.domain.informative_features.InformativeFeaturesCalculator import BaseSelector


class ExternalPredictorEvaluation:
    @classmethod
    def execute(cls, selectors_class: List[Type[BaseSelector]], splitted_dataset: SplittedDataset, config: DatasetConfig) -> None:
        Logger.execute("Metric: External Predictor Evaluation")
        test_dataset = splitted_dataset.get_test()
        train_dataset = splitted_dataset.get_train()
        feature_names = splitted_dataset.get_feature_names()
        for selector_class in selectors_class:
            ranking_per_specificity = RankingReader.execute(selector_class, config)
            for specificity in ranking_per_specificity.keys():
                ranking_list = ranking_per_specificity[specificity]
                if specificity == "general":
                    Logger.execute(f"-- Label: {specificity}")
                elif specificity.startswith("label"):
                    label = int(specificity.replace('label', ''))
                    Logger.execute(f"-- Label: {label}")
                    
                else:
                    Logger.execute(f"-- [ERROR] Specificity {specificity} is invalid!")
            
    def _evaluate_selected_features(splitted_dataset: SplittedDataset, feature_names: List[str], ranking_list: List[pd.DataFrame], config: DatasetConfig) -> None:
        X_train = splitted_dataset.get_train().get_features()
        y_train = splitted_dataset.get_train().get_labels()
        X_test = splitted_dataset.get_test().get_features()
        y_test = splitted_dataset.get_test().get_labels()
        for i in range(0, config.external_predictor_k):
            number_of_features = i + 1
            Logger.execute(f"--- Number of features: {number_of_features}")
            accuracy_list = []
            for ranking in ranking_list:
                selected_features = ranking['feature'].to_numpy()[0:number_of_features]
                selected_features_index = [feature_names.index(feature_name) for feature_name in selected_features]
                print(selected_features)
                print(selected_features_index)
                X_train_selected = X_train[:, selected_features_index]
                X_test_selected = X_test[:, selected_features_index]
                svc_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=config.random_seed)
                svc_model.fit(X_train_selected, y_train)
                y_pred = svc_model.predict(X_test_selected)
                report = ClassificationReportCalculator.execute(y_test, y_pred, splitted_dataset.get_n_labels())
                
            Logger.execute(f"---- Accuracy: {accuracy_list}")
            Logger.execute(f"----- Mean: {statistics.mean(accuracy_list)}")
            Logger.execute(f"----- Standard deviation: {statistics.stdev(accuracy_list)}")
            
        