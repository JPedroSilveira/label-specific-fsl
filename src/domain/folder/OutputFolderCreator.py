from os import path
from src.util.file_util import create_folder_if_not_exists
from config.type import OutputConfig

class OutputFolderCreator:
    @staticmethod
    def execute(output_config: OutputConfig, execution_id: str) -> None:      
        output_config.temporary = path.join(output_config.root, output_config.temporary)
        output_config.execution_output.root = path.join(output_config.root, execution_id)
        output_config.execution_output.execution_time = path.join(output_config.execution_output.root, output_config.execution_output.execution_time)
        output_config.execution_output.feature_erasure = path.join(output_config.execution_output.root, output_config.execution_output.feature_erasure)
        output_config.execution_output.informative_features = path.join(output_config.execution_output.root, output_config.execution_output.informative_features)
        output_config.execution_output.predictive_performance = path.join(output_config.execution_output.root, output_config.execution_output.predictive_performance)
        output_config.execution_output.selection_performance = path.join(output_config.execution_output.root, output_config.execution_output.selection_performance)
        output_config.execution_output.stability = path.join(output_config.execution_output.root, output_config.execution_output.stability)
        output_config.execution_output.weighted_tsne = path.join(output_config.execution_output.root, output_config.execution_output.weighted_tsne)
        output_config.execution_output.raw_selection = path.join(output_config.execution_output.root, output_config.execution_output.raw_selection)
        create_folder_if_not_exists(output_config.root)
        create_folder_if_not_exists(output_config.temporary)
        create_folder_if_not_exists(output_config.execution_output.root)
        create_folder_if_not_exists(output_config.execution_output.execution_time)
        create_folder_if_not_exists(output_config.execution_output.feature_erasure)
        create_folder_if_not_exists(output_config.execution_output.informative_features)
        create_folder_if_not_exists(output_config.execution_output.predictive_performance)
        create_folder_if_not_exists(output_config.execution_output.selection_performance)
        create_folder_if_not_exists(output_config.execution_output.stability)
        create_folder_if_not_exists(output_config.execution_output.weighted_tsne)
        create_folder_if_not_exists(output_config.execution_output.raw_selection)
    