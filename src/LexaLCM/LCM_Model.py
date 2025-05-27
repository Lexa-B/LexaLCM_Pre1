# src/LexaLCM/LCM_Model.py

import torch
from torch.amp import autocast
from transformers import PreTrainedModel, MODEL_MAPPING
from LexaLCM.LCM_Config import LexaLCMConfig

Verbose = True

class LexaLCM(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # PreNet - Contextualizer -> Normalize and map input SONAR embeddings to the model's hidden dimension. fp32 -> bf16
        self.PreNet_C_Up = torch.nn.Linear(config.input_dim, config.d_model).to(torch.float32)
        
        # Contextualizer
        self.Contextualizer_Across = torch.nn.Linear(config.d_model, config.d_model).to(torch.bfloat16)
        
        # # Contextualizer - RoPE. (fp32)
        # self.Contextualizer_RoPE = torch.nn.Linear(config.d_latent, config.d_latent).to(torch.float32)

        # PostNet - Contextualizer -> Normalize and reduce from the model's hidden dimension to the latent dimension.
        self.PostNet_C_Down = torch.nn.Linear(config.d_model, config.d_latent).to(torch.bfloat16)

        # Latent Layer
        self.Latent_Layer = torch.nn.Linear(config.d_latent, config.d_latent).to(torch.bfloat16)

        # PreNet - Denoiser -> Normalize and map the latent dimension to the model's hidden dimension.
        self.PreNet_D_Up = torch.nn.Linear(config.d_latent, config.d_model).to(torch.bfloat16)

        # Denoiser
        self.Denoiser_Across = torch.nn.Linear(config.d_model, config.d_model).to(torch.bfloat16)

        # PostNet - Denoiser -> Normalize and reduce from the model's hidden dimension to the input dimension. bf16 -> fp32
        self.PostNet_D_Down = torch.nn.Linear(config.d_model, config.input_dim).to(torch.float32)

    def forward(self, embeddings, labels=None):
        # Input: [batch_size=1, seq_len=3, input_dim=1024]
        if Verbose:
            print(f"[DEBUG - model] embeddings: shape={embeddings.shape}, dtype={embeddings.dtype}")
        #x = embeddings

        # PreNet - Contextualizer
        with autocast(device_type="cuda", enabled=False):
            x = embeddings.to(torch.float32)
            x = self.PreNet_C_Up(x)
        if Verbose:
            print(f"[DEBUG - model] after PreNet_C_Up: shape={x.shape}, dtype={x.dtype}")
        x = x.to(torch.bfloat16)

        # Contextualizer
        if Verbose:
            print(f"[DEBUG - model] Contextualizer_Across: shape={x.shape}, dtype={x.dtype}")
        x = self.Contextualizer_Across(x)
        if Verbose:
            print(f"[DEBUG - model] after Contextualizer_Across: shape={x.shape}, dtype={x.dtype}")

        # PostNet - Contextualizer
        if Verbose:
            print(f"[DEBUG - model] PostNet_C_Down: shape={x.shape}, dtype={x.dtype}")
        x = self.PostNet_C_Down(x)
        if Verbose:
            print(f"[DEBUG - model] after PostNet_C_Down: shape={x.shape}, dtype={x.dtype}")

        # Latent Layer
        if Verbose:
            print(f"[DEBUG - model] Latent_Layer: shape={x.shape}, dtype={x.dtype}")
        x = self.Latent_Layer(x)
        if Verbose:
            print(f"[DEBUG - model] after Latent_Layer: shape={x.shape}, dtype={x.dtype}")

        # PreNet - Denoiser
        if Verbose:
            print(f"[DEBUG - model] PreNet_D_Up: shape={x.shape}, dtype={x.dtype}")
        x = self.PreNet_D_Up(x)
        if Verbose:
            print(f"[DEBUG - model] after PreNet_D_Up: shape={x.shape}, dtype={x.dtype}")

        # Denoiser
        if Verbose:
            print(f"[DEBUG - model] Denoiser_Across: shape={x.shape}, dtype={x.dtype}")
        x = self.Denoiser_Across(x)
        if Verbose:
            print(f"[DEBUG - model] after Denoiser_Across: shape={x.shape}, dtype={x.dtype}")
        
        # PostNet - Denoiser
        with autocast(device_type="cuda", enabled=False):  # Force out of AMP
            x = x.to(torch.float32)
            if Verbose:
                print(f"[DEBUG - model] after to(float32) PostNet_D_Down: shape={x.shape}, dtype={x.dtype}")
            x = self.PostNet_D_Down(x)  # fp32
            if Verbose:
                print(f"[DEBUG - model] after PostNet_D_Down: shape={x.shape}, dtype={x.dtype}")
        x = x.squeeze(1)  # remove sequence dimension to get [batch_size=1, input_dim]
        if Verbose:
            print(f"[DEBUG - model] final output: shape={x.shape}, dtype={x.dtype}")
        
        loss = None
        if labels is not None:
            loss = torch.mean((x - labels) ** 2)
        
        return {"loss": loss, "logits": x}

MODEL_MAPPING.register(LexaLCMConfig, LexaLCM)