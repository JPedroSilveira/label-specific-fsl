from cProfile import label


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
    ignored_columns: list[str]
    test_percentage: int
    k_fold: int
    k_fold_repeat: int

class FeatureConfig:
    informative_prefix: str
    informative_per_label_prefix: str

class Config:
    output: OutputConfig
    dataset: DatasetConfig
    feature: FeatureConfig
    random_seed: int

    
        