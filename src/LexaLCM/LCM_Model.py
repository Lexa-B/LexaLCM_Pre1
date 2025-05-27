# src/LexaLCM/LCM_Model.py

import torch
from torch.amp import autocast
from transformers import PreTrainedModel, MODEL_MAPPING
from LexaLCM.LCM_Config import LexaLCMConfig

class LexaLCM(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        
        # Tower 1: fp32 -> bf16
        self.tower1_up = torch.nn.Linear(config.input_dim, config.d_model).to(torch.float32)
        self.tower1_down = torch.nn.Linear(config.d_model, config.d_latent).to(torch.bfloat16)
        
        # Tower 2: bf16 -> fp32
        self.tower2_up = torch.nn.Linear(config.d_latent, config.d_model).to(torch.bfloat16)
        self.tower2_down = torch.nn.Linear(config.d_model, config.input_dim).to(torch.float32)

    def forward(self, embeddings, labels=None):
        # Input: [batch_size=1, seq_len=3, input_dim=1024]
        print(f"[DEBUG - model] embeddings: shape={embeddings.shape}, dtype={embeddings.dtype}")
        
        # Tower 1 (process each embedding)
        x = embeddings
        x = self.tower1_up(x)  # fp32
        print(f"[DEBUG - model] after tower1_up: shape={x.shape}, dtype={x.dtype}")
        x = x.to(torch.bfloat16)
        print(f"[DEBUG - model] after to(bfloat16): shape={x.shape}, dtype={x.dtype}")
        x = self.tower1_down(x)  # bf16
        print(f"[DEBUG - model] after tower1_down: shape={x.shape}, dtype={x.dtype}")
        
        # Aggregate to single vector (mean pooling)
        x = x.mean(dim=1, keepdim=True)  # [batch_size=1, 1, d_latent], bf16
        print(f"[DEBUG - model] after mean pooling: shape={x.shape}, dtype={x.dtype}")
        
        # Tower 2
        x = self.tower2_up(x)  # bf16
        print(f"[DEBUG - model] after tower2_up: shape={x.shape}, dtype={x.dtype}")
        with autocast(device_type="cuda", enabled=False):  # Force out of AMP
            x = x.to(torch.float32)
            print(f"[DEBUG - model] after to(float32) before tower2_down: shape={x.shape}, dtype={x.dtype}")
            x = self.tower2_down(x)  # fp32
            print(f"[DEBUG - model] after tower2_down: shape={x.shape}, dtype={x.dtype}")
        x = x.squeeze(1)  # remove sequence dimension to get [batch_size=1, input_dim]
        print(f"[DEBUG - model] final output: shape={x.shape}, dtype={x.dtype}")
        
        loss = None
        if labels is not None:
            loss = torch.mean((x - labels) ** 2)
        
        return {"loss": loss, "logits": x}

MODEL_MAPPING.register(LexaLCMConfig, LexaLCM)