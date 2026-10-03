"""
Fashion-MNIST Classifier in PyTorch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_fashion_mnist
import tempfile
from pathlib import Path
import requests
import gzip
import torch

def load_fashion_mnist(n_train=10000, n_test=2000):
    loaded_data = {}
    
    BASE_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets"
    FILE_LIST = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]
    LABEL_FLAGS = [False, True, False, True]
    DICT_KEYS = ['X_train', 'y_train', 'X_test', 'y_test']
    DATA_COUNTS = [n_train, n_train, n_test, n_test]

    for filename, label_flag, dict_key, data_count in zip(FILE_LIST, LABEL_FLAGS, DICT_KEYS, DATA_COUNTS):
        file_path = Path(tempfile.gettempdir()) / filename
        
        # download the four idx gz files once
        if not file_path.is_file():
            response = requests.get(f"{BASE_URL}/{filename}", timeout=60)
            response.raise_for_status()
            file_path.write_bytes(response.content)

        # parse with np.frombuffer
        with gzip.open(file_path, "rb") as f:
            data_bytes = f.read()
            data = np.frombuffer(data_bytes, dtype=np.uint8, offset=8 if label_flag else 16)
            
            if label_flag:
                data = data[:data_count]
                torch_tensor = torch.from_numpy(data.copy()).to(dtype=torch.int64)
            else:
                data = data.reshape(-1, 28, 28).copy()
                data = data[:data_count]
                torch_tensor = torch.from_numpy(data.copy()).to(dtype=torch.float32) / 255.0
            
            loaded_data[dict_key] = torch_tensor

    # return float tensors in 0-1 and int64 labels.
    return loaded_data

# Step 2 - FashionDataset
class FashionDataset(Dataset):
    def __init__(self, X, y, mean=0.2860, std=0.3530):
        self.X = X
        self.m, _, _ = self.X.shape
        self.y = y
        self.mean = mean
        self.std = std

    def __len__(self):
        return self.m

    def __getitem__(self, i):
        return ((self.X[i] - self.mean) / self.std, self.y[i])

# Step 3 - make_loaders
from torch.utils.data import TensorDataset, DataLoader

def make_loaders(data, batch_size=64, val_size=2000, seed=42):
    train_size = data['X_train'].size(0) - val_size
    
    # extract tensors for training and validation
    X_train, X_val = torch.split(data['X_train'], [train_size, val_size], dim=0)
    y_train, y_val = torch.split(data['y_train'], [train_size, val_size], dim=0)

    # shuffle training data
    generator = torch.Generator().manual_seed(seed)
    shuffled_indices = torch.randperm(train_size, generator=generator)
    X_train = X_train[shuffled_indices]
    y_train = y_train[shuffled_indices]

    # prepare loaders
    loaders = {}
    loaders['train'] = DataLoader(TensorDataset(X_train, y_train), batch_size)
    loaders['val'] = DataLoader(TensorDataset(X_val, y_val), batch_size)
    loaders['test'] = DataLoader(TensorDataset(data['X_test'], data['y_test']), batch_size)
    loaders['sizes'] = (train_size, val_size, data['y_test'].size(0))

    return loaders

# Step 4 - MLP
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, hidden1=300, hidden2=100, n_classes=10):
        super().__init__()
        # fc1 (784 -> hidden1)
        self.fc1 = nn.Linear(784, hidden1)
        # fc2 (hidden1 -> hidden2)
        self.fc2 = nn.Linear(hidden1, hidden2)
        # out (hidden2 -> n_classes)
        self.out = nn.Linear(hidden2, n_classes)

    def forward(self, x):
        # flatten, ReLU after fc1 and fc2, return logits
        x = torch.flatten(x, start_dim=1) # preserve batch dim
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.out(x)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Step 5 - train_one_epoch
def train_one_epoch(model, loader, loss_fn, optimizer):
    model.train()

    # metrics
    total_loss = 0.0
    total_samples = 0

    for inputs, labels in loader:
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = loss_fn(predictions, labels)

        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples

# Step 6 - evaluate (not yet solved)
# TODO: implement

# Step 7 - fit (not yet solved)
# TODO: implement

# Step 8 - lr_range_test (not yet solved)
# TODO: implement

# Step 9 - random_search (not yet solved)
# TODO: implement

# Step 10 - test_accuracy (not yet solved)
# TODO: implement

# Step 11 - save_model (not yet solved)
# TODO: implement

# Step 12 - predict_classes (not yet solved)
# TODO: implement

