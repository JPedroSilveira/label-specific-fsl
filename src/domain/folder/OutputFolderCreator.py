from os import path
import os
from config.type import OutputConfig

class OutputFolderCreator:
    @staticmethod
    def execute(output_config: OutputConfig, execution_id: str) -> None:      
        output_config.temporary = path.join(output_config.root, output_config.temporary)
        output_config.execution_output.root = path.join(output_config.root, execution_id)
        output_config.execution_output.log_file = path.join(output_config.execution_output.root, output_config.execution_output.log_file)
        output_config.execution_output.execution_time = path.join(output_config.execution_output.root, output_config.execution_output.execution_time)
        output_config.execution_output.feature_erasure = path.join(output_config.execution_output.root, output_config.execution_output.feature_erasure)
        output_config.execution_output.informative_features = path.join(output_config.execution_output.root, output_config.execution_output.informative_features)
        output_config.execution_output.predictive_performance = path.join(output_config.execution_output.root, output_config.execution_output.predictive_performance)
        output_config.execution_output.selection_performance = path.join(output_config.execution_output.root, output_config.execution_output.selection_performance)
        output_config.execution_output.stability = path.join(output_config.execution_output.root, output_config.execution_output.stability)
        output_config.execution_output.weighted_tsne = path.join(output_config.execution_output.root, output_config.execution_output.weighted_tsne)
        output_config.execution_output.raw_selection = path.join(output_config.execution_output.root, output_config.execution_output.raw_selection)
        OutputFolderCreator._create_folder_if_not_exists(output_config.root)
        OutputFolderCreator._create_folder_if_not_exists(output_config.temporary)
        OutputFolderCreator._create_folder_if_not_exists(output_config.execution_output.root)
        OutputFolderCreator._create_folder_if_not_exists(output_config.execution_output.execution_time)
        OutputFolderCreator._create_folder_if_not_exists(output_config.execution_output.feature_erasure)
        OutputFolderCreator._create_folder_if_not_exists(output_config.execution_output.informative_features)
        OutputFolderCreator._create_folder_if_not_exists(output_config.execution_output.predictive_performance)
        OutputFolderCreator._create_folder_if_not_exists(output_config.execution_output.selection_performance)
        OutputFolderCreator._create_folder_if_not_exists(output_config.execution_output.stability)
        OutputFolderCreator._create_folder_if_not_exists(output_config.execution_output.weighted_tsne)
        OutputFolderCreator._create_folder_if_not_exists(output_config.execution_output.raw_selection)
            
    @staticmethod
    def _create_folder_if_not_exists(folder_path) -> None:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
    