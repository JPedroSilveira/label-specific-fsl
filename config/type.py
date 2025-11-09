from typing import List


class ExecutionOutputConfig:
    root: str
    selection_performance: str
    feature_erasure: str
    predictive_performance: str
    weighted_tsne: str
    stability: str
    execution_time: str
    informative_features: str
    raw_selection: str

class OutputConfig:
    root: str
    temporary: str
    execution_output: ExecutionOutputConfig
    
class DatasetConfig:
    root: str
    filename: str
    label_column: str
    ignored_columns: List[str]
    test_percentage: int
    k_fold: int
    k_fold_repeat: int
    regularization_lambda: float
    lime_k: int
    shap_k: int
    shap_representative_k: int
    relieff_k: int
    lasso_regularization: float
    data_type: str
    label_type: str
    batch_size: int
    epochs: int
    learning_rate: float
    model: int

class FeatureConfig:
    informative_prefix: str
    informative_per_label_prefix: str
    
class SelectorConfig:
    model: str

class Config:
    output: OutputConfig
    dataset: DatasetConfig
    feature: FeatureConfig
    random_seed: int
    selectors: List[SelectorConfig]

    
        