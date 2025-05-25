import os
import glob
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

class SonarDataset(Dataset):
    def __init__(self, data_dir, split, text_column):
        self.text_column = text_column
        self.index = []

        # Point to the correct split directory (e.g., data_dir/train/)
        split_dir = os.path.join(data_dir, split)
        files = sorted(glob.glob(os.path.join(split_dir, "*.parquet")))

        for path in files:
            try:
                df = pd.read_parquet(path, columns=[text_column])
                for i in range(len(df)):
                    self.index.append((path, i))
            except Exception as e:
                print(f"⚠️ Skipping file {path}: {e}")

        print(f"📂 Indexed {len(self.index)} {split} samples from {len(files)} files")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        path, row_idx = self.index[idx]
        df = pd.read_parquet(path, columns=[self.text_column])
        row = df.iloc[row_idx][self.text_column]

        try:
            arr = np.stack([np.array(vec, dtype=np.float32) for vec in row])
            if len(arr.shape) != 2:
                raise ValueError(f"Expected 2D embedding array, got: {arr.shape}")
            if arr.shape[1] != 1024:
                raise ValueError(f"Expected embedding dimension: 1024, got: {arr.shape[1]}")
            return torch.tensor(arr)
        except Exception as e:
            print(f"❌ Skipping corrupted embedding at {path}:{row_idx} → {e}")
            return torch.zeros((1, 1024), dtype=torch.float32)
