from typing import List
import imageio
import uuid
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from src.config.general_config import TEMP_OUTPUT_PATH
from src.util.performance_util import ExecutionTimeCounter


def save_plots_as_video(file_name: str, plot_list: List[Figure], fps=1):
    """
    Saves a list of Matplotlib plots as a video.
    """
    execution_time_counter = ExecutionTimeCounter().start()
    print(f"Persisting plots as video...")
    images = []
    temp_file_name = f"{TEMP_OUTPUT_PATH}/{uuid.uuid4()}-temp.png"
    for plot in plot_list:
        plot.savefig(temp_file_name)
        images.append(imageio.imread(temp_file_name))
        plot.clear()
        plt.close(plot)
    imageio.mimsave(file_name, images, fps=fps, format="GIF")
    execution_time_counter.print_end('Video creation')