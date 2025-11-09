class ExecutionOutputFolders:
    selection_performance: str
    feature_erasure: str
    predictive_performance: str
    weighted_tsne: str
    stability: str
    execution_time: str
    informative_features: str
    raw_selection: str

class Output:
    general: str
    temporary: str
    execution_output: str
    execution_output_folders: ExecutionOutputFolders

class Config:
    output: Output

    
        