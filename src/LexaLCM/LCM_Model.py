# src/LexaLCM/LCM_Model.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.amp import autocast
from transformers import PreTrainedModel, MODEL_MAPPING
from LexaLCM.LCM_Config import LexaLCMConfig

Verbose = True


class NormalizeInput(nn.Module): # ToDo: add input normalization as per the Meta FAIR paper
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
    
    def forward(self, x):
        return x

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))  # learnable scaling

    def forward(self, x):
        # Calculate root mean square: sqrt(mean(x_i^2))
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()
        # Normalize and scale
        return self.weight * (x / (rms + self.eps))
    
class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float):
        super().__init__()
        self.norm = RMSNorm(features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # Pre-normalize input → pass through sublayer → dropout → residual add
        return x + self.dropout(sublayer(self.norm(x)))

class FeedForward_SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff * 2)  # Note: d_ff * 2!
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x_proj = self.linear1(x)  # shape: (batch, seq, d_ff * 2)
        x_gated, x_linear = x_proj.chunk(2, dim=-1)  # Split into two halves
        x_act = F.silu(x_gated) * x_linear           # SwiGLU activation
        x_drop = self.dropout(x_act)
        return self.linear2(x_drop)
    
def apply_rope_to(q, k):
    # q, k: (batch, heads, seq, dim)
    seq_len = q.shape[2]
    dim = q.shape[-1]
    device = q.device

    # Create RoPE frequencies
    position = torch.arange(seq_len, device=device).unsqueeze(1)
    dim_idx = torch.arange(0, dim, 2, device=device)
    inv_freq = 1.0 / (10000 ** (dim_idx / dim))
    freqs = position * inv_freq  # (seq, dim/2)

    sin = freqs.sin().unsqueeze(0).unsqueeze(0)  # (1, 1, seq, dim/2)
    cos = freqs.cos().unsqueeze(0).unsqueeze(0)

    def rotate(x):
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        x_rot = torch.stack([
            x1 * cos - x2 * sin,
            x2 * cos + x1 * sin
        ], dim=-1)
        return x_rot.flatten(-2)

    return rotate(q), rotate(k)

def causal_mask(seq_len: int, device):
    return torch.tril(torch.ones((1, 1, seq_len, seq_len), device=device)).bool()
    
class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model # Embedding vector size
        self.h = h # Number of heads
        # Make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False) # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # Just apply the formula from the paper
        # (batch, h, seq, d_k) --> (batch, h, seq, seq)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq, seq) # Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq, seq) --> (batch, h, seq, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch, seq, d_model) --> (batch, seq, d_model)
        key = self.w_k(k) # (batch, seq, d_model) --> (batch, seq, d_model)
        value = self.w_v(v) # (batch, seq, d_model) --> (batch, seq, d_model)

        # (batch, seq, d_model) --> (batch, seq, h, d_k) --> (batch, h, seq, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # Combine all the heads together
        # (batch, h, seq, d_k) --> (batch, seq, h, d_k) --> (batch, seq, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by Wo
        # (batch, seq, d_model) --> (batch, seq, d_model)  
        return self.w_o(x)

class GeneralAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout, use_rope=False):
        super().__init__()
        self.use_rope = use_rope
        self.core = MultiHeadAttentionBlock(d_model, n_heads, dropout)

    def forward(self, q, k, v, mask=None):
        if self.use_rope:
            q, k = apply_rope_to(q, k)
        return self.core(q, k, v, mask)
    
class ContextualSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.attn = GeneralAttention(d_model, n_heads, dropout, use_rope=True)

    def forward(self, x):
        mask = causal_mask(x.shape[1], x.device)
        return self.attn(x, x, x, mask)

class DenoiserSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.attn = GeneralAttention(d_model, n_heads, dropout, use_rope=False)

    def forward(self, x):
        mask = torch.eye(x.shape[1], device=x.device).unsqueeze(0).unsqueeze(0)
        return self.attn(x, x, x, mask)

class DenoiserCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.attn = GeneralAttention(d_model, n_heads, dropout, use_rope=False)

    def forward(self, x, context):
        return self.attn(x, context, context)
















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