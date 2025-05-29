# src/LexaLCM/Data/DataHandler.py

from safetensors import safe_open
from torch.utils.data import Dataset  
from transformers import DataCollator
import torch
import os
import glob
import pandas as pd
import numpy as np

class LCMDataset(Dataset):
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

class LCMDataset(Dataset):
    def __init__(self, data_dir, split, text_column, max_seq_len=None):
        self.text_column = text_column
        self.index = []
        self.max_seq_len = max_seq_len

        # Load SoT and EoT once
        with safe_open("src/LexaLCM/Data/SpecialConcepts/StartOfText.safetensors", framework="pt") as f:
            self.sot = f.get_tensor("embedding").squeeze()

        with safe_open("src/LexaLCM/Data/SpecialConcepts/EndOfText.safetensors", framework="pt") as f:
            self.eot = f.get_tensor("embedding").squeeze()

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
            arr = [np.array(vec, dtype=np.float32) for vec in row]
            if any(x.shape != (1024,) for x in arr):
                raise ValueError("Non-1024 embedding detected.")

            tensor = torch.tensor(np.stack(arr))  # [seq, 1024]

            # Prepend SoT and append EoT
            tensor = torch.cat([
                self.sot.unsqueeze(0),
                tensor,
                self.eot.unsqueeze(0)
            ], dim=0)  # [seq + 2, 1024]

            # Truncate if too long
            if self.max_seq_len and tensor.shape[0] > self.max_seq_len:
                tensor = tensor[:self.max_seq_len]

            return {
                "embeddings": tensor,
                "labels": tensor[-1]  # Optional: adjust if you want last non-pad token
            }

        except Exception as e:
            print(f"❌ Skipping corrupted embedding at {path}:{row_idx} → {e}")
            zero_tensor = torch.zeros((1, 1024), dtype=torch.float32)
            return {
                "embeddings": zero_tensor,
                "labels": zero_tensor.squeeze(0)
            }

class LCMDataset_DryRun(Dataset):
    def __init__(self):
        self.samples = []
        # Load 3 embeddings as a single training example
        for name in ["StartOfText", "HelloWorld", "EndOfText"]:
            with safe_open(f"src/LexaLCM/Data/SpecialConcepts/{name}.safetensors", framework="pt") as f:
                embedding = f.get_tensor("embedding").squeeze(-2)  # [1024]
                self.samples.append(embedding)
        
        # Each training example is a sequence of 3 embeddings: SoT, "Hello world.", EoT
        self.input_sequence = torch.stack(self.samples)  # [3, 1024]
        self.target = self.samples[-1]  # [1024] (last embedding)

    def __len__(self):
        return 100  # Mock 100 samples for testing batching

    def __getitem__(self, idx):
        # Return the same example repeatedly for testing
        return {
            "embeddings": self.input_sequence,  # [3, 1024]
            "labels": self.target               # [1024]
        }
    
class LCMCollator:
    def __init__(self):
        # Load the padding embedding: expected shape [1024] or [1, 1024]
        with safe_open("src/LexaLCM/Data/SpecialConcepts/PaddingSentence.safetensors", framework="pt") as f:
            pad_tensor = f.get_tensor("embedding")
            self.pad_embedding = pad_tensor.squeeze()  # Ensure shape is [1024]

    def __call__(self, features):
        # Each 'embeddings' entry is [seq, 1024]
        sequences = [f["embeddings"] for f in features]
        labels = torch.stack([f["labels"] for f in features])  # [batch, 1024]

        max_len = max(seq.shape[0] for seq in sequences)

        padded = []
        for seq in sequences:
            pad_len = max_len - seq.shape[0]
            if pad_len > 0:
                pad = self.pad_embedding.unsqueeze(0).repeat(pad_len, 1)  # [pad_len, 1024]
                seq = torch.cat([seq, pad], dim=0)  # [max_len, 1024]
            padded.append(seq)

        batch_embeddings = torch.stack(padded)  # [batch, max_len, 1024]

        attention_masks = []
        for seq in sequences:
            seq_len = seq.shape[0]
            pad_len = max_len - seq_len
            mask = torch.ones(seq_len, dtype=torch.bool)
            if pad_len > 0:
                pad = torch.zeros(pad_len, dtype=torch.bool)
                mask = torch.cat([mask, pad])
            attention_masks.append(mask)

        batch_attention_mask = torch.stack(attention_masks)  # [batch, max_len]

        return {
            "embeddings": batch_embeddings,        # [batch, seq, 1024]
            "labels": labels,                      # [batch, 1024]
            "attention_mask": batch_attention_mask # [batch, seq]
        }
