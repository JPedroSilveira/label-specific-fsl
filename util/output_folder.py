from os import path
from util.file_util import create_folder_if_not_exists
from config.config import Output

def create_output_files(output_config: Output, execution_id: str):
    output_config.execution_output = path.join(output_config.general, execution_id)
    output_config.execution_output_folders.execution_time = path.join(output_config.execution_output, output_config.execution_output_folders.execution_time)
    output_config.execution_output_folders.feature_erasure = path.join(output_config.execution_output, output_config.execution_output_folders.feature_erasure)
    output_config.execution_output_folders.informative_features = path.join(output_config.execution_output, output_config.execution_output_folders.informative_features)
    output_config.execution_output_folders.predictive_performance = path.join(output_config.execution_output, output_config.execution_output_folders.predictive_performance)
    output_config.execution_output_folders.selection_performance = path.join(output_config.execution_output, output_config.execution_output_folders.selection_performance)
    output_config.execution_output_folders.stability = path.join(output_config.execution_output, output_config.execution_output_folders.stability)
    output_config.execution_output_folders.weighted_tsne = path.join(output_config.execution_output, output_config.execution_output_folders.weighted_tsne)
    output_config.execution_output_folders.raw_selection = path.join(output_config.execution_output, output_config.execution_output_folders.raw_selection)
    create_folder_if_not_exists(output_config.general)
    create_folder_if_not_exists(output_config.temporary)
    create_folder_if_not_exists(output_config.execution_output)
    create_folder_if_not_exists(output_config.execution_output_folders.execution_time)
    create_folder_if_not_exists(output_config.execution_output_folders.feature_erasure)
    create_folder_if_not_exists(output_config.execution_output_folders.informative_features)
    create_folder_if_not_exists(output_config.execution_output_folders.predictive_performance)
    create_folder_if_not_exists(output_config.execution_output_folders.selection_performance)
    create_folder_if_not_exists(output_config.execution_output_folders.stability)
    create_folder_if_not_exists(output_config.execution_output_folders.weighted_tsne)
    create_folder_if_not_exists(output_config.execution_output_folders.raw_selection)
    