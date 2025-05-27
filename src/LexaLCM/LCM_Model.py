# src/LexaLCM/LCM_Model.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.amp import autocast
from transformers import PreTrainedModel, MODEL_MAPPING
from LexaLCM.LCM_Config import LexaLCMConfig

# ToDo: make this a global variable that can be set to True/False from the command line
Verbose = True

## ------------------------------------------------------------
## Helper Layers
## ------------------------------------------------------------

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
        self.weight = nn.Parameter(torch.ones(d_model))  # starts as float32

    def forward(self, x):
        # If needed, cast weight to x's dtype and device for safe mixed precision
        weight = self.weight.to(dtype=x.dtype, device=x.device)
        
        # Compute root mean square
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()

        return weight * (x / (rms + self.eps))

    
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
    
# RoPE-related functions and storage
def generate_sin_cos(seq_len, dim, device):
    half_dim = dim // 2
    inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim, device=device).float() / half_dim))
    positions = torch.arange(seq_len, device=device).float()
    sinusoid_inp = torch.einsum("i,j->ij", positions, inv_freq)
    sin = torch.sin(sinusoid_inp)
    cos = torch.cos(sinusoid_inp)
    sin = torch.cat([sin, sin], dim=-1)  # expand to full dim
    cos = torch.cat([cos, cos], dim=-1)
    sin = sin.to(torch.float32)
    cos = cos.to(torch.float32)
    if Verbose:
        if sin.dtype != torch.float32 or cos.dtype != torch.float32:
            print(f"[WARN] RoPE sin/cos dtype not float32! sin: {sin.dtype}, cos: {cos.dtype}")
        else:
            print(f"[DEBUG - model] RoPE sin/cos dtype: {sin.dtype}, {cos.dtype}")
    return sin.unsqueeze(0), cos.unsqueeze(0)  # [1, seq_len, dim]

def rotate(x):
    x1 = x[..., ::2]  # even dims
    x2 = x[..., 1::2]  # odd dims
    return torch.stack([-x2, x1], dim=-1).reshape_as(x)

def apply_rope_to(q, k, sin, cos):
    # Ensure sin/cos are same dtype and shape as q/k
    sin = sin[:, :q.shape[1], :].to(dtype=q.dtype, device=q.device)
    cos = cos[:, :q.shape[1], :].to(dtype=q.dtype, device=q.device)
    q_rot = q * cos + rotate(q) * sin
    k_rot = k * cos + rotate(k) * sin
    return q_rot, k_rot

def causal_mask(size, device):
    return torch.tril(torch.ones(size, size, device=device)).unsqueeze(0).unsqueeze(0)
    
## Attention Blocks

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"
        self.d_k = d_model // h

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        return self.w_o(x)

class GeneralAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout, use_rope=False):
        super().__init__()
        self.use_rope = use_rope
        self.core = MultiHeadAttentionBlock(d_model, n_heads, dropout)

    def forward(self, q, k, v, mask=None):
        if self.use_rope:
            sin, cos = generate_sin_cos(seq_len=q.size(1), dim=q.size(-1), device=q.device)
            q, k = apply_rope_to(q, k, sin, cos)
            if Verbose:
                print(f"[DEBUG - model] RoPE sin/cos dtype in attention block: {sin.dtype}, {cos.dtype}")
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

## ------------------------------------------------------------
## Main Layers
## ------------------------------------------------------------

## PreNets and PostNets

class PreNetC(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, out_dim)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.norm(x)
        x = self.proj(x)
        x = self.act(x)
        return x

class PostNetC(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.proj(x)

class PreNetD(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, out_dim)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.norm(x)
        x = self.proj(x)
        x = self.act(x)
        return x

class PostNetD(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.proj(x)

## Contextualizer Tower

class ContextualizerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attention = ContextualSelfAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward_SwiGLU(d_model, d_ff, dropout)

        self.residual_connections = nn.ModuleList([
            ResidualConnection(d_model, dropout),
            ResidualConnection(d_model, dropout)
        ])

    def forward(self, x):
        x = self.residual_connections[0](x, self.self_attention)
        x = self.residual_connections[1](x, self.feed_forward)
        return x

class ContextualizerTower(nn.Module):
    def __init__(self, num_layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([
            ContextualizerLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)  # Final norm layer (post-residual stack)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if Verbose:
                print(f"[DEBUG - model] Before ContextualizerLayer {i}: dtype = {x.dtype}")
            x = layer(x)
            if Verbose:
                print(f"[DEBUG - model] After ContextualizerLayer {i}: dtype = {x.dtype}")
        x = self.norm(x)
        if Verbose:
            print(f"[DEBUG - model] After ContextualizerTower norm (before dtype clamp): dtype = {x.dtype}")
        x = x.to(torch.bfloat16) # Clamp back to bf16 bacause the fp32 RMS value causes it to be promoted to fp32
        if Verbose:
            print(f"[DEBUG - model] After ContextualizerTower norm: dtype = {x.dtype}")
        return x

## Latent Bridge

class LatentBridge(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            RMSNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.proj(x)

## Denoiser Tower
























## ------------------------------------------------------------
## LexaLCM Model's Main Architecture
## ------------------------------------------------------------

class LexaLCM(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.PreNet_C_Up = PreNetC(config.input_dim, config.d_model)

        self.ContextualizerTower = ContextualizerTower(
            num_layers=config.num_context_layers,
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            dropout=config.dropout
        )

        self.PostNet_C_Down = PostNetC(config.d_model, config.d_latent)

        self.LatentBridge = LatentBridge(config.d_latent, dropout=config.dropout)

        self.PreNet_D_Up = PreNetD(config.d_latent, config.d_model)

        self.Denoiser_Across = torch.nn.Linear(config.d_model, config.d_model)

        self.PostNet_D_Down = PostNetD(config.d_model, config.input_dim)

    def forward(self, embeddings, labels=None):
        if Verbose:
            print(f"[DEBUG - model] embeddings: shape={embeddings.shape}, dtype={embeddings.dtype}")

        x = self.PreNet_C_Up(embeddings)
        if Verbose:
            print(f"[DEBUG - model] after PreNet_C_Up: shape={x.shape}, dtype={x.dtype}")

        x = self.ContextualizerTower(x)
        if Verbose:
            print(f"[DEBUG - model] after ContextualizerTower: shape={x.shape}, dtype={x.dtype}")

        x = self.PostNet_C_Down(x)
        if Verbose:
            print(f"[DEBUG - model] after PostNet_C_Down: shape={x.shape}, dtype={x.dtype}")

        x = self.LatentBridge(x)
        if Verbose:
            print(f"[DEBUG - model] after LatentBridge: shape={x.shape}, dtype={x.dtype}")

        x = self.PreNet_D_Up(x)
        if Verbose:
            print(f"[DEBUG - model] after PreNet_D_Up: shape={x.shape}, dtype={x.dtype}")

        x = self.Denoiser_Across(x)
        if Verbose:
            print(f"[DEBUG - model] after Denoiser_Across: shape={x.shape}, dtype={x.dtype}")

        with autocast(device_type="cuda", enabled=False):
            x = x.to(torch.float32)
            if Verbose:
                print(f"[DEBUG - model] after to(float32) PostNet_D_Down: shape={x.shape}, dtype={x.dtype}")
            x = self.PostNet_D_Down(x)
            if Verbose:
                print(f"[DEBUG - model] after PostNet_D_Down: shape={x.shape}, dtype={x.dtype}")

        x = x.squeeze(1)
        if Verbose:
            print(f"[DEBUG - model] final output: shape={x.shape}, dtype={x.dtype}")

        loss = None
        if labels is not None:
            loss = torch.mean((x - labels) ** 2)

        return {"loss": loss, "logits": x}

MODEL_MAPPING.register(LexaLCMConfig, LexaLCM)