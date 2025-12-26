import os
from typing import Any, Dict, List
from matplotlib import pyplot as plt

from src.domain.prediction.types.ClassificationScoreAggregatedReport import ClassificationScoreAggregatedReport


class ClassificationReportAggregatedGraph:
    @classmethod
    def execute(cls, selector_name: str, ranking_label: str, aggregated_report_per_number_of_features: Dict[int, ClassificationScoreAggregatedReport], output_folder: str) -> None:
        sorted_keys = sorted(aggregated_report_per_number_of_features.keys())
        general_metrics = {
            #"Accuracy": {"avg": [], "std": []},
            "F1-Score": {"avg": [], "std": []},
            #"Precision": {"avg": [], "std": []},
            #"Recall":   {"avg": [], "std": []}
        }
        metrics_per_label: Dict[int, Dict[str, Dict[str, List[float]]]] = {}
        for k in sorted_keys:
            report = aggregated_report_per_number_of_features[k]
            general = report.general
            #general_metrics["Accuracy"]["avg"].append(general.accuracy_avg)
            #general_metrics["Accuracy"]["std"].append(general.accuracy_stdev)
            general_metrics["F1-Score"]["avg"].append(general.f1_score_avg)
            general_metrics["F1-Score"]["std"].append(general.f1_score_stdev)
            #general_metrics["Precision"]["avg"].append(general.precision_avg)
            #general_metrics["Precision"]["std"].append(general.precision_stdev)
            #general_metrics["Recall"]["avg"].append(general.recall_avg)
            #general_metrics["Recall"]["std"].append(general.recall_stdev)
            for label_report in report.per_label:
                label = label_report.label
                if label not in metrics_per_label:
                    metrics_per_label[label] = {
                        "F1-Score": {"avg": [], "std": []},
                        #"Precision": {"avg": [], "std": []},
                        #"Recall":    {"avg": [], "std": []}
                    }
                metrics_per_label[label]["F1-Score"]["avg"].append(label_report.f1_score_avg)
                metrics_per_label[label]["F1-Score"]["std"].append(label_report.f1_score_stdev)
                #metrics_per_label[label]["Precision"]["avg"].append(label_report.precision_avg)
                #metrics_per_label[label]["Precision"]["std"].append(label_report.precision_stdev)
                #metrics_per_label[label]["Recall"]["avg"].append(label_report.recall_avg)
                #metrics_per_label[label]["Recall"]["std"].append(label_report.recall_stdev)
        cls._plot_evolution(f"{selector_name.lower()}_{ranking_label}_ranking_effect_over_general_prediction", sorted_keys, general_metrics, output_folder)
        for label, label_metrics in metrics_per_label.items():
            cls._plot_evolution(f"{selector_name.lower()}_{ranking_label}_ranking_effect_over_{label}_prediction", sorted_keys, label_metrics, output_folder)
    
    @staticmethod
    def _plot_evolution(filename: str, x_values: List[int], metrics_dict: Dict[str, Any], output_folder: str) -> None:
        plt.figure(figsize=(10, 6))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # Standard cycle colors
        for i, (metric_name, values) in enumerate(metrics_dict.items()):
            avg = values["avg"]
            std = values["std"]
            color = colors[i % len(colors)]
            plt.plot(x_values, avg, marker='o', label=metric_name, color=color)
            lower_bound = [max(0, a - s) for a, s in zip(avg, std)]
            upper_bound = [min(1, a + s) for a, s in zip(avg, std)]
            plt.fill_between(x_values, lower_bound, upper_bound, color=color, alpha=0.15)
        plt.xlabel("Number of Features Removed")
        plt.ylabel("Score")
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, filename + ".pdf"), dpi=300, format='pdf')
        plt.close()