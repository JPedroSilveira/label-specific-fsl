import numpy as np
from pandas import DataFrame
from sklearn import metrics
from sklearn.manifold import TSNE
from typing import Dict, List, Type
import matplotlib.pyplot as plt

from config.type import Config
from src.domain.data.Normalizer import Normalizer
from src.domain.data.types.Dataset import Dataset 
from src.domain.data.reader.WeightReader import WeightReader
from src.domain.selector.types.base.BaseSelector import BaseSelector
from src.domain.log.Logger import Logger


class WTSNECreator:
    @classmethod
    def execute(cls, selectors_class: List[Type[BaseSelector]], dataset: Dataset, config: Config) -> None:
        silhouette_per_selector_and_specificity: Dict[str, Dict[str,List[float]]] = {}
        embedding_silhouette = cls._calculate_wtsne(0, "without_weights", None, None, dataset, config)
        Logger.execute(f"TSNE: {embedding_silhouette}")
        silhouette_per_selector_and_specificity["None"] = { "None": [ embedding_silhouette ]}
        weights_per_selector_and_specificity: Dict[str, Dict[str,List[DataFrame]]] = {}
        for selector_class in selectors_class:
            weights_per_selector_and_specificity[selector_class.get_name()] = WeightReader.execute(selector_class, config)
        Logger.execute(f"WTSNE:")
        for selector_class in selectors_class:
                weights_per_selector_and_specificity[selector_class.get_name] = { }
                weights_per_specificity = weights_per_selector_and_specificity[selector_class.get_name()]
                Logger.execute(f"- Selector: {selector_class.get_name()}")
                for specificity in weights_per_specificity.keys():
                    weights_per_selector_and_specificity[selector_class.get_name][specificity] = []
                    weights = weights_per_specificity[specificity]
                    embedding_silhouettes = []
                    for execution, execution_weights in enumerate(weights):
                        embedding_silhouette = cls._calculate_wtsne(execution, specificity, selector_class, execution_weights, dataset, config)
                        weights_per_selector_and_specificity[selector_class.get_name][specificity].append(embedding_silhouette)
                        embedding_silhouettes.append(embedding_silhouette)
                    Logger.execute(f"-- Label {specificity}: {embedding_silhouettes}")    
                    
    @staticmethod
    def _calculate_wtsne(execution: int, specificity: str, selector_class: BaseSelector, weights: DataFrame, dataset: Dataset, config: Config) -> float:     
        X = dataset.get_features().copy()
        y = dataset.get_labels().copy()
        title = f"TSNE"
        if weights is not None:
            title = f"Weighted TSNE"
            weights = weights['value'].to_numpy()
            W = Normalizer.execute(weights)
            X = X * np.sqrt(W)
        perplexity = max(30, len(X)/100)
        learning_rate = max(200, dataset.get_n_features()/12) / 4
        Logger.execute(f"-- Perplexity: {perplexity}")   
        Logger.execute(f"-- Learning rate: {learning_rate}")   
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            max_iter=500,
            verbose=1,
            random_state=config.dataset.random_seed,
            init='pca',
            metric='euclidean',
            learning_rate=learning_rate
        )
        embedding = tsne.fit_transform(X)
        embedding_silhouette = metrics.silhouette_score(embedding, y, metric='euclidean')
        if selector_class is None:
            datapath = f'{config.output.execution_output.weighted_tsne}/tsne.pdf'
        else:
            datapath = f'{config.output.execution_output.weighted_tsne}/wtsne_{selector_class.get_name()}_{specificity}_{execution}.pdf'
        colors = [
            '#FF0000',  # Pure Red
            '#0000FF',  # Pure Blue
            '#008000',  # Green
            '#FFA500',  # Orange
            '#800080',  # Purple
            '#00FFFF',  # Cyan
            '#FFC0CB',  # Pink
            '#A52A2A',  # Brown
            '#808000',  # Olive
            '#000000'   # Black
        ]
        markers = ['o', 's', '^', 'D', 'v', '>', '<', 'p', '*', 'X']
        plt.figure(figsize=(8, 6))
        for i, label in enumerate(np.unique(y)):
            idx = y == label
            plt.scatter(
                embedding[idx, 0],
                embedding[idx, 1],
                c=colors[i % len(colors)],
                marker=markers[i % len(markers)],
                alpha=0.6,
                label=label
            )
        #plt.title(title)
        #plt.xlabel("t-SNE 1")
        #plt.ylabel("t-SNE 2")
        plt.legend(title="Classes", loc='best') # Add the legend after all scatter calls
        
        # --- Add Title and Labels (Uncommented for a complete chart) ---
        #plt.title(title)
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.grid(True)
        plt.savefig(datapath)
        plt.close()
        return embedding_silhouette