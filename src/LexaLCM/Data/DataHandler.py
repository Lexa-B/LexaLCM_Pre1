# src/LexaLCM/Data/DataHandler.py

from safetensors import safe_open
from torch.utils.data import Dataset  
from transformers import DataCollator
import torch


class LCMDataset(Dataset):
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
    def __call__(self, features):
        return {
            # Stack sequences: (batch_size, seq_len=3, input_dim=1024)
            "embeddings": torch.stack([f["embeddings"] for f in features]),
            # Stack labels: (batch_size, input_dim=1024)
            "labels": torch.stack([f["labels"] for f in features])
        }