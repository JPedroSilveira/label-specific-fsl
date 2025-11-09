import torch
import numpy as np
from torch import nn
from tqdm import tqdm

from config.type import DatasetConfig
from src.device import device
from src.domain.pytorch.PyTorchDataLoaderCreator import PyTorchDataLoaderCreator

from sklearn.utils.class_weight import compute_class_weight


class PyTorchSimpleFit:
    @staticmethod
    def execute(model: nn.Module, X: np.ndarray, y: np.ndarray, config: DatasetConfig) -> None:
        # Define loss criterion
        criterion = PyTorchSimpleFit._get_criterion(y)
        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        # Setup train mode
        model.train()
        # Create data loader
        data_loader = PyTorchDataLoaderCreator.execute(X, y, config)
        # Info variables
        loss = None
        # Create an outer loop progress bar for epochs
        epoch_iterator = tqdm(range(config.epochs), desc="Epoch Progress")
        # Train
        for _ in epoch_iterator: 
            for Xd, yd in data_loader:
                # Reset optimizer
                optimizer.zero_grad()
                # Forward pass
                y_prediction = model(Xd)
                # Computeh Loss
                loss = criterion(y_prediction, yd)
                # Backpropagation
                loss.backward()
                # Update weights
                optimizer.step()
            epoch_iterator.set_postfix(
                epoch_loss=f"{loss.item():.8f}"
            )
        epoch_iterator.close()
        
    @staticmethod
    def _get_criterion(y: np.ndarray) -> nn.CrossEntropyLoss:
        # Labels
        labels = np.unique(y)
        # Compute label weight
        label_weights=compute_class_weight(class_weight="balanced", classes=labels, y=y)
        label_weights=torch.tensor(label_weights, dtype=torch.float).to(device)
        # Define loss criterion
        cross_entropy_loss = nn.CrossEntropyLoss(weight=label_weights)
        # One hot encoder
        # Criterion method
        return cross_entropy_loss