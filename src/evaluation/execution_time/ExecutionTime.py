import csv
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from src.config.general_config import OUTPUT_PATH, EXECUTION_TIME_OUTPUT_SUB_PATH

class ExecutionTimeStatistics:
    def __init__(self, execution_times):
        self.average = np.mean(execution_times)
        self.variance = np.var(execution_times)

def create_execution_time_table_and_chart(execution_times_by_selector: dict[str, list[float]]):
    _create_execution_average_table(execution_times_by_selector)
    _create_execution_time_chart(execution_times_by_selector)

def _calculate_execution_time_statistics(execution_times_by_selector: dict[str, list[float]]):
    '''
    Given the history of a selector executions calculate the average execution time
    '''
    execution_time_statistics_by_selector: dict[str, ExecutionTimeStatistics] = {}
    for selector, execution_times in execution_times_by_src.selector.items():
        statistics = ExecutionTimeStatistics(execution_times)
        execution_time_statistics_by_selector[selector] = statistics
    return execution_time_statistics_by_selector

def _create_execution_average_table(execution_times_by_selector: dict[str, list[float]]):
    execution_time_statistics_by_selector: dict[str, ExecutionTimeStatistics] = _calculate_execution_time_statistics(execution_times_by_selector)
    data = []
    for selector, statistics in execution_time_statistics_by_src.selector.items():
        data.append([selector, round(statistics.average, 3), round(statistics.variance, 3)])
    # Sort items by execution time
    data = sorted(data, key=lambda x: x[1], reverse=True)
    # Add header
    data.insert(0, ["Algorithm", "Average (s)", "Variance (s)"])
    # Persist using LaTEX format
    with open(f'{OUTPUT_PATH}/{EXECUTION_TIME_OUTPUT_SUB_PATH}/average-execution-time.txt', "w") as file:
        file.write(tabulate(data, headers="firstrow", tablefmt="latex"))
    # Persist using HTML format
    with open(f'{OUTPUT_PATH}/{EXECUTION_TIME_OUTPUT_SUB_PATH}/average-execution-time.html', "w") as file:
        file.write(tabulate(data, headers="firstrow", tablefmt="html"))
    # Persist using CSV format
    with open(f'{OUTPUT_PATH}/{EXECUTION_TIME_OUTPUT_SUB_PATH}/average-execution-time.csv', "w", newline="") as file:
        csvwriter = csv.writer(file)
        csvwriter.writerow(data[0])  
        csvwriter.writerows(data[1:])

def _create_execution_time_chart(execution_times_by_selector: dict[str, list[float]]):
    '''
    Given a dictionary with the execution times for each selector, persist a boxplot chart
    '''
    # Extract data for plotting
    data = [execution_times_by_selector[label] for label in execution_times_by_selector]
    labels = list(execution_times_by_src.selector.keys())
    # Set figure dimensions
    plt.figure()
    # Create the boxplot
    plt.boxplot(data)
    # Set figure title and legends
    plt.title('')
    plt.ylabel('Time (seconds)')
    #plt.xlabel('Feature selection algorithm')
    plt.xticks(range(1, len(labels) + 1), labels)
    # Persist chart as image
    plt.savefig(f'{OUTPUT_PATH}/{EXECUTION_TIME_OUTPUT_SUB_PATH}/execution-time-distribution.pdf', format='pdf', bbox_inches='tight')
    # Reset plt
    plt.close()