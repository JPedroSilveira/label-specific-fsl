import torch
import random
import numpy as np
from torch import nn
from tqdm import tqdm

from config.type import DatasetConfig
from src.domain.device.DeviceGetter import DeviceGetter
from src.domain.pytorch.PyTorchPerLabelDataLoaderCreator import PyTorchPerLabelDataLoaderCreator
from src.domain.data.types.Dataset import Dataset

from sklearn.utils.class_weight import compute_class_weight

class PyTorchPerLabelFit:
    @classmethod
    def execute(cls, model: nn.Module, train_dataset: Dataset, config: DatasetConfig) -> None:
        # Enable regularization
        enable_regularization = callable(getattr(model, "get_regularization", None))
        # Enable before forward
        enable_before_forward = callable(getattr(model, "before_forward", None))
        # Enable after forward
        enable_after_forward = callable(getattr(model, "after_forward", None))
        # Get train features and labels
        features = train_dataset.get_features()
        labels = train_dataset.get_labels()
        # Define loss criterion
        criterion = cls._get_criterion(labels)
        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        # Info variables
        reg = 0.0
        loss_without_reg = 0.0
        # Create an outer loop progress bar for epochs
        epoch_iterator = tqdm(range(config.epochs), desc="Epoch Progress")
        # Train
        for _ in epoch_iterator: 
            # Create train data loaders by label
            by_label_data_loaders = PyTorchPerLabelDataLoaderCreator.execute(features, labels, config)
            # Define iterators
            iterators = []
            for i in range(0, len(by_label_data_loaders)):
                iterators.append(iter(by_label_data_loaders[i]))
            # Setup train mode
            model.train()
            # Iterate over data loaders for each class
            for _ in range(0, len(by_label_data_loaders[0])):
                for i in cls._shuffed_range(0, len(by_label_data_loaders)):
                    try:
                        X, y = next(iterators[i])
                        # Send expected output to model when necessary
                        if enable_before_forward:
                            model.before_forward(y, train_dataset.get_label_types())
                        # Forward pass
                        y_prediction = model(X)
                        # Computeh Loss
                        loss = criterion(y_prediction, y)
                        loss_without_reg = loss.item()
                        if enable_regularization:
                            reg = model.get_regularization()
                            loss += reg
                        # Backpropagation
                        loss.backward()
                        # Update weights
                        optimizer.step()   
                        # Reset optimizer
                        optimizer.zero_grad() 
                        if enable_after_forward:
                            model.after_forward()
                    except StopIteration:
                        continue
            epoch_iterator.set_postfix(
                epoch_loss=f"{loss_without_reg:.8f}", 
                reg=f"{reg:.8f}"
            )
        epoch_iterator.close()

    @staticmethod
    def _get_criterion(y: np.ndarray) -> nn.CrossEntropyLoss:
        # Labels
        labels = np.unique(y)
        # Compute label weight
        label_weights=compute_class_weight(class_weight="balanced", classes=labels, y=y)
        label_weights=torch.tensor(label_weights, dtype=torch.float).to(DeviceGetter.execute())
        # Define loss criterion
        cross_entropy_loss = nn.CrossEntropyLoss(weight=label_weights)
        # One hot encoder
        # Criterion method
        return cross_entropy_loss

    @staticmethod
    def _shuffed_range(min, max) -> list[int]:
        values = list(range(min, max))
        random.shuffle(values)
        return values
