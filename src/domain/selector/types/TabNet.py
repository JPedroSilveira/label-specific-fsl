import torch
import numpy as np
import pandas as pd
from pytorch_tabnet.tab_model import TabNetClassifier

from config.type import DatasetConfig
from src.domain.pytorch.PyTorchFit import PyTorchFit
from src.domain.device.DeviceGetter import DeviceGetter
from src.domain.selector.types.DecisionTree import BaseSelectorWeight, Dataset, SelectorSpecificity   


class TabNet(BaseSelectorWeight):
    def __init__(self, n_features: int, n_labels: int, config: DatasetConfig) -> None:
        super().__init__(n_features, n_labels, config)
        self._model = TabNetClassifier(
            optimizer_fn=torch.optim.AdamW,
            optimizer_params=dict(lr=0.001),
            n_d=8, # From 8 to 64
            n_a=8, # Same as n_d
            n_steps=5, # From 3 to 10
            gamma=1.0, # From 1.0 to 2.0 
            n_independent=1, # From 1 to 5
            #n_shared=2, # Default 2 
            seed=config.random_seed,
            momentum=0.02, # From 0.01 to 0.4
            lambda_sparse=1e-5,
            #mask_type="sparsemax" # Default sparsemax 
        )

    def get_name() -> str:
        return "TabNet"
    
    def can_predict(self) -> bool:
        return True
    
    def get_specificity(self) -> SelectorSpecificity:
        return SelectorSpecificity.PER_LABEL
    
    def fit(self, train_dataset: Dataset, test_dataset: Dataset) -> None: 
        self._model.fit(train_dataset.get_features(), train_dataset.get_labels(), max_epochs=1000, 
                        loss_fn=PyTorchFit._get_criterion(train_dataset.get_labels()), 
                        eval_set=[(test_dataset.get_features(), test_dataset.get_labels())], 
                        eval_metric=['logloss'],
                        batch_size=16, 
                        patience=50)
        feature_names = train_dataset.get_feature_names()
        explain_matrix, masks = self._model.explain(train_dataset.get_features())
        importance_df = pd.DataFrame(explain_matrix, columns=feature_names)
        importance_df['target_class'] = np.array(train_dataset.get_labels()).flatten()
        self._per_class_importance = importance_df.groupby('target_class').mean().to_numpy()

    def predict(self, dataset: Dataset) -> np.ndarray:
        return self._model.predict(dataset.get_features())
    
    def predict_probabilities(self, dataset: Dataset, use_softmax: bool=True) -> np.ndarray:
        return self._model.predict_proba(dataset.get_features())
    
    def get_general_weights(self) -> np.ndarray:
        return self._model.feature_importances_
    
    def get_per_label_weights(self) -> np.ndarray:
        return self._per_class_importance