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

# Step 3 - make_loaders (not yet solved)
# TODO: implement

# Step 4 - MLP (not yet solved)
# TODO: implement

# Step 5 - train_one_epoch (not yet solved)
# TODO: implement

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

