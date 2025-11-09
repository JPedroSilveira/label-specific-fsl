import torch
import random
import numpy as np
from torch import nn
from src.data.Dataset import Dataset
from src.pytorch_helpers.PyTorchDataLoader import get_by_label_data_loaders, get_data_loader
from src.util.print_util import print_load_bar
from src.config.general_config import SHOULD_LOG_WHILE_TRAINING, USE_MODEL_REGULARIZATION, EPOCHS, USE_SGD_OPTIMIZER, DEFAULT_LEARNING_RATE
from sklearn.utils.class_weight import compute_class_weight
from src.util.device_util import get_device

def pytorch_simple_fit(model: nn.Module, X: np.ndarray, y: np.ndarray, n_epochs: int = EPOCHS, lr: int = DEFAULT_LEARNING_RATE):
    # Define loss criterion
    criterion = _get_criterion(y)
    # Create optimizer
    optimizer = _get_optimizer(model, lr)
    # Setup train mode
    src.model.train()
    # Create data loader
    data_loader = get_data_loader(X, y)
    # Train
    for epoch in range(n_epochs): 
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
        # print_load_bar(epoch+1, n_epochs)

def pytorch_fit(model: nn.Module, train_dataset: Dataset, n_epochs: int = EPOCHS, lr: int = DEFAULT_LEARNING_RATE):
    # Enable regularization
    enable_regularization = USE_MODEL_REGULARIZATION and callable(getattr(model, "get_regularization", None))
    # Enable before forward
    enable_before_forward = callable(getattr(model, "before_forward", None))
    # Enable after forward
    enable_after_forward = callable(getattr(model, "after_forward", None))
    # Get train/validation features and labels
    train_features = train_dataset.get_features()
    train_labels = train_dataset.get_labels()
    # Define loss criterion
    criterion = _get_criterion(train_labels)
    # Create optimizer
    optimizer = _get_optimizer(model, lr) 
    # Create train data loader
    train_data_loader = get_data_loader(train_features, train_labels)
    # Info variables
    reg = 0.0
    loss_without_reg = 0.0
    # Train        
    for epoch in range(n_epochs): 
        # Setup train mode
        src.model.train()
        for X, y in train_data_loader:
            # Send expected output to model when necessary
            if enable_before_forward:
                src.model.before_forward(y, train_dataset.get_label_types())
            # Forward pass
            y_prediction = model(X)
            # Computeh Loss
            loss = criterion(y_prediction, y)
            if enable_regularization:
                loss_without_reg = loss.item()
                reg = src.model.get_regularization()
                loss += reg
            else:
                loss_without_reg = loss.item()
            # Backpropagation
            loss.backward()
            # Update weights
            optimizer.step()
            # Reset optimizer
            optimizer.zero_grad() 
            if enable_after_forward:
                src.model.after_forward()
        if SHOULD_LOG_WHILE_TRAINING:
            print_load_bar(epoch+1, n_epochs, info=f"Loss: {loss_without_reg:.8f} | Reg: {reg:.8f}")

def pytorch_fit_by_label(model: nn.Module, train_dataset: Dataset, test_dataset: Dataset, n_epochs: int = EPOCHS, lr: int = DEFAULT_LEARNING_RATE):
    # Enable regularization
    enable_regularization = USE_MODEL_REGULARIZATION and callable(getattr(model, "get_regularization", None))
    # Enable before forward
    enable_before_forward = callable(getattr(model, "before_forward", None))
    # Enable after forward
    enable_after_forward = callable(getattr(model, "after_forward", None))
    # Get train features and labels
    features = train_dataset.get_features()
    labels = train_dataset.get_labels()
    # Define loss criterion
    criterion = _get_criterion(labels)
    # Create optimizer
    optimizer = _get_optimizer(model, lr) 
    # Train
    for epoch in range(n_epochs): 
        # Create train data loaders by label
        by_label_data_loaders = get_by_label_data_loaders(features, labels)
        # Define iterators
        iterators = []
        for i in range(0, len(by_label_data_loaders)):
          iterators.append(iter(by_label_data_loaders[i]))
        # Setup train mode
        src.model.train()
        # Persist current regularization output
        reg = 0.0
        # Iterate over data loaders for each class
        for _ in range(0, len(by_label_data_loaders[0])):
            for i in _shuffed_range(0, len(by_label_data_loaders)):
                try:
                    X, y = next(iterators[i])
                    # Send expected output to model when necessary
                    if enable_before_forward:
                        src.model.before_forward(y, train_dataset.get_label_types())
                    # Forward pass
                    y_prediction = model(X)
                    # Computeh Loss
                    loss = criterion(y_prediction, y)
                    if enable_regularization:
                        reg = src.model.get_regularization()
                        loss += reg
                    # Backpropagation
                    loss.backward()
                    # Update weights
                    optimizer.step()   
                    # Reset optimizer
                    optimizer.zero_grad() 
                    if enable_after_forward:
                        src.model.after_forward()
                except StopIteration:
                    continue
        if SHOULD_LOG_WHILE_TRAINING:
            print_load_bar(epoch+1, n_epochs)

def _get_optimizer(model: nn.Module, lr: float):
    # Create optimizer
    optimizer = None 
    if USE_SGD_OPTIMIZER:
        optimizer = torch.optim.SGD(src.model.parameters(), lr=lr, momentum=0.9) 
    else:
        optimizer = torch.optim.AdamW(src.model.parameters(), lr=lr)
    return optimizer

def _get_criterion(y: np.ndarray):
    device = get_device()
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

def _shuffed_range(min, max):
    values = list(range(min, max))
    random.shuffle(values)
    return values
