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

        files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
        for path in files:
            df = pd.read_parquet(path, columns=["split"])
            matching = df["split"] == split
            indices = matching[matching].index.tolist()
            self.index.extend([(path, i) for i in indices])

        print(f"📂 Indexed {len(self.index)} {split} samples from {len(files)} files")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        path, row_idx = self.index[idx]
        df = pd.read_parquet(path, columns=[self.text_column])
        row = df.iloc[row_idx][self.text_column]

        try:
            arr = np.array(row, dtype=np.float32)
            if len(arr.shape) != 2:
                raise ValueError(f"Invalid shape: {arr.shape}")
            return torch.tensor(arr)
        except Exception as e:
            print(f"❌ Skipping corrupted embedding at {path}:{row_idx} → {e}")
            # Return a dummy tensor so training doesn’t crash
            return torch.zeros((1, 1024), dtype=torch.float32)



