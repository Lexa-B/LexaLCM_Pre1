# src/LexaLCM/Data/DataHandler.py

from safetensors import safe_open
from torch.utils.data import Dataset  
from transformers import DataCollator
import torch
import os
import glob
import pandas as pd
import numpy as np
import pyarrow.parquet as pq

class LCMDataset(Dataset):
    def __init__(self, data_dir, split, text_column, max_seq_len=None, cache_size=10, sample_size=None):
        self.text_column = text_column
        self.max_seq_len = max_seq_len
        self.cache_size = cache_size  # Number of files to keep in memory
        self.data_cache = {}  # LRU cache for parquet files
        self.file_paths = []  # Just store file paths
        self.file_sizes = {}  # Store file sizes for length calculation
        self.total_samples = 0
        self.indices = []

        # Load SoT and EoT once
        with safe_open("src/LexaLCM/Data/SpecialConcepts/StartOfText.safetensors", framework="pt") as f:
            self.sot = f.get_tensor("embedding").squeeze()

        with safe_open("src/LexaLCM/Data/SpecialConcepts/EndOfText.safetensors", framework="pt") as f:
            self.eot = f.get_tensor("embedding").squeeze()

        # Just collect file paths
        split_dir = os.path.join(data_dir, split)
        self.file_paths = sorted(glob.glob(os.path.join(split_dir, "*.parquet")))
        print(f"📂 Found {len(self.file_paths)} parquet files")

        # Pre-calculate total samples using parquet metadata
        print("📊 Calculating dataset size...")
        for path in self.file_paths:
            try:
                # Use pyarrow to get row count without loading data
                parquet_file = pq.ParquetFile(path)
                num_rows = parquet_file.metadata.num_rows
                self.file_sizes[path] = num_rows
                self.total_samples += num_rows
            except Exception as e:
                print(f"⚠️ Error reading {path}: {e}")
                # If metadata reading fails, fall back to loading the file
                df = pd.read_parquet(path, columns=[self.text_column])
                self.file_sizes[path] = len(df)
                self.total_samples += len(df)
        
        print(f"📊 Total samples: {self.total_samples:,}")
 
        if sample_size and sample_size < self.total_samples:
            print(f"🎲 Sampling {sample_size} out of {self.total_samples} total samples.")
            self.indices = np.random.choice(self.total_samples, sample_size, replace=False).tolist()
        else:
            self.indices = list(range(self.total_samples))

    def _get_file_and_row(self, idx):
        """Convert global index to file and row index"""
        if idx >= self.total_samples:
            raise IndexError(f"Index {idx} out of range (total: {self.total_samples})")
            
        current_idx = 0
        for path in self.file_paths:
            file_size = self.file_sizes[path]
            if current_idx + file_size > idx:
                return path, idx - current_idx
            current_idx += file_size
        
        raise IndexError("Index out of range")

    def _load_file(self, path):
        """Load a file into cache, maintaining cache size limit"""
        if path in self.data_cache:
            return self.data_cache[path]
        
        # If cache is full, remove oldest entry
        if len(self.data_cache) >= self.cache_size:
            oldest_key = next(iter(self.data_cache))
            del self.data_cache[oldest_key]
        
        # Load new file
        df = pd.read_parquet(path, columns=[self.text_column])
        self.data_cache[path] = df
        return df

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]  # use the sampled index
        path, row_idx = self._get_file_and_row(actual_idx)
        
        df = self._load_file(path)
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
                "labels": tensor[-1]
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
