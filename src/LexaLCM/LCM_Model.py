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

        if Verbose:
            print(f"[DEBUG - model] RMSNorm Input dtype: x = {x.dtype}")

        # If needed, cast weight to x's dtype and device for safe mixed precision
        weight = self.weight.to(dtype=x.dtype, device=x.device)
        
        # Compute root mean square
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()

        if Verbose:
            print(f"[DEBUG - model] RMSNorm Output dtype: x = {x.dtype}, rms = {rms.dtype}")

        return weight * (x / (rms + self.eps))

## AdaLN 

class TimestepEmbedder(nn.Module):
    def __init__(self, d_model, t_emb_dim):
        super().__init__()
        self.t_emb_dim = t_emb_dim
        self.freq_embedding_dim = t_emb_dim

        # Frequency embedding: sinusoidal
        self.lin1 = nn.Linear(t_emb_dim, t_emb_dim)
        self.act = nn.SiLU()
        self.lin2 = nn.Linear(t_emb_dim, d_model)

    def forward(self, timestep):  # timestep: [B, 1, 1] or scalar
        device = timestep.device
        half_dim = self.freq_embedding_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half_dim, dtype=torch.float32, device=device) / half_dim
        )
        # timestep: [B, 1, 1] → [B]
        t = timestep.view(-1).float()  # Ensures t is always [B]
        sinusoid = torch.outer(t, freqs)
        emb = torch.cat([sinusoid.sin(), sinusoid.cos()], dim=-1)  # shape: [B, t_emb_dim]
        # Feed through MLP
        emb = self.lin2(self.act(self.lin1(emb)))
        return emb  # shape: [B, d_model]

class AdaLNModulator(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.ff = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, d_model * 3),  # γ, β, α
        )
        self.init_zero()

    def init_zero(self):
        # Start as identity function
        for layer in self.ff:
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, t_emb):
        return self.ff(t_emb).chunk(3, dim=-1)  # returns γ, β, α

## Other

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
        if Verbose:
            print(f"[DEBUG - model] FeedForward_SwiGLU Input dtype: x = {x.dtype}")
        x_proj = self.linear1(x)  # shape: (batch, seq, d_ff * 2)
        x_gated, x_linear = x_proj.chunk(2, dim=-1)  # Split into two halves
        x_act = F.silu(x_gated) * x_linear           # SwiGLU activation
        x_drop = self.dropout(x_act)
        if Verbose:
            print(f"[DEBUG - model] FeedForward_SwiGLU Output dtype: x_drop = {x_drop.dtype}")
        return self.linear2(x_drop)

class FeedForward_AdaLN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.modulator = AdaLNModulator(d_model)
        self.linear1 = nn.Linear(d_model, d_ff * 2)  # SwiGLU: needs 2x d_ff
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x, t_emb):

        if Verbose:
            print(f"[DEBUG - model] FeedForward_AdaLN Input dtype: x = {x.dtype}")

        # Step 1: Compute modulation params
        γ, β, α = self.modulator(t_emb)  # Each [B, D]
        γ = γ.unsqueeze(1)  # [B, 1, D]
        β = β.unsqueeze(1)
        α = α.unsqueeze(1)

        # Step 2: Modulate input
        x_mod = (1 + γ) * x + β  # [B, T, D]

        # Step 3: SwiGLU MLP
        x_proj = self.linear1(x_mod)
        x_gated, x_linear = x_proj.chunk(2, dim=-1)
        x_act = F.silu(x_gated) * x_linear
        x_out = self.linear2(self.dropout(x_act))

        if Verbose:
            print(f"[DEBUG - model] FeedForward_AdaLN Output dtype: x = {x.dtype}, α = {α.dtype}, x_out = {x_out.dtype}")

        # Step 4: Residual with AdaLN α gate
        return x + α * x_out
        
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
        if Verbose:
            print(f"[DEBUG - model] ContextualSelfAttention Input dtype: x = {x.dtype}")
        mask = causal_mask(x.shape[1], x.device)
        if Verbose:
            print(f"[DEBUG - model] ContextualSelfAttention Mask dtype: x = {x.dtype}, mask = {mask.dtype}")
        return self.attn(x, x, x, mask)

class DenoiserSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.attn = GeneralAttention(d_model, n_heads, dropout, use_rope=False)
        self.modulator = AdaLNModulator(d_model)

    def forward(self, x, t_emb):

        if Verbose:
            print(f"[DEBUG - model] DenoiserSelfAttention Input dtype: x = {x.dtype}")

        # 1. Modulate input with AdaLN based on timestep
        γ, β, α = self.modulator(t_emb) # [batch, d_model] each
        γ = γ.unsqueeze(1) # -> [batch, 1, d_model]
        β = β.unsqueeze(1) # -> [batch, 1, d_model]
        α = α.unsqueeze(1) # -> [batch, 1, d_model]

        # 2. Apply AdaLN modulation
        x_mod = (1 + γ) * x + β

        # 3. Create causal mask [1, 1, seq_len, seq_len]
        seq_len = x.shape[1]
        mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device)).unsqueeze(0).unsqueeze(1)

        # 4. Run attention (Q = K = V = x_mod)
        y = self.attn(x_mod, x_mod, x_mod, mask)

        if Verbose:
            print(f"[DEBUG - model] DenoiserSelfAttention Output dtype: x = {x.dtype}, α = {α.dtype}, y = {y.dtype}")

        # 5. Apply residual connection and scaling
        return x + α * y
    
class DenoiserCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.attn = GeneralAttention(d_model, n_heads, dropout, use_rope=False)
        self.modulator = AdaLNModulator(d_model)

    def forward(self, x, context, t_emb, *, dropout_denoiser=0.0, training=False):

        if Verbose:
            print(f"[DEBUG - model] DenoiserCrossAttention Input dtype: x = {x.dtype}")

        # 1. Modulate input with AdaLN based on timestep
        γ, β, α = self.modulator(t_emb) # [batch, d_model] each
        γ = γ.unsqueeze(1) # -> [batch, 1, d_model]
        β = β.unsqueeze(1) # -> [batch, 1, d_model]
        α = α.unsqueeze(1) # -> [batch, 1, d_model]

        # 2. Apply AdaLN modulation
        x_mod = (1 + γ) * x + β

        # 3. Prepend zero-vector to context, which provides the position 0 something to attend to
        zero = torch.zeros((context.size(0), 1, context.size(2)), device=context.device, dtype=context.dtype)
        context = torch.cat([zero, context], dim=1)

        # 4. Build causal mask for the context sequence
        seq_len_q = x_mod.size(1)
        seq_len_k = context.size(1)
        causal_mask = torch.tril(torch.ones((seq_len_q, seq_len_k), device=x.device)).unsqueeze(0).unsqueeze(1)

        # 5. Apply Row-Level CFG Dropout
        if training and dropout_denoiser > 0.0:
            keep_mask = (torch.rand(context.size(0), context.size(1), device=context.device) > dropout_denoiser).float()
            keep_mask[:, 0] = 1.0
            context = context * keep_mask.unsqueeze(-1)
            causal_mask = causal_mask * keep_mask.unsqueeze(1).unsqueeze(2)

        # 6. Run attention
        y = self.attn(x_mod, context, context, causal_mask)

        if Verbose:
            print(f"[DEBUG - model] DenoiserCrossAttention Output dtype: x = {x.dtype}, α = {α.dtype}, y = {y.dtype}")

        # 7. Apply residual connection and scaling
        return x + α * y


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
        self.mlp = FeedForward_SwiGLU(d_model, d_ff, dropout)

        self.residual_connections = nn.ModuleList([
            ResidualConnection(d_model, dropout),
            ResidualConnection(d_model, dropout)
        ])

    def forward(self, x):
        x = self.residual_connections[0](x, self.self_attention)
        x = self.residual_connections[1](x, self.mlp)
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

class DenoiserLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_heads: int, dropout: float):
        super().__init__()
        self.self_attention = DenoiserSelfAttention(d_model, n_heads, dropout)
        self.cross_attention = DenoiserCrossAttention(d_model, n_heads, dropout)
        self.mlp = FeedForward_AdaLN(d_model, d_ff, dropout)

    def forward(self, x, context, timestep, *, dropout_denoiser=0.0, training=False):
        # Each sublayer already handles AdaLN inside
        x = x + self.self_attention(x, timestep)
        x = x + self.cross_attention(x, context, timestep, dropout_denoiser=dropout_denoiser, training=training)
        x = x + self.mlp(x, timestep)
        return x

class DenoiserTower(nn.Module):
    def __init__(self, num_layers, d_model, d_ff, n_heads, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            DenoiserLayer(d_model, d_ff, n_heads, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model)  # Optional — depends on paper interpretation

    def forward(self, x, context, timestep, *, dropout_denoiser=0.0, training=False):
        with autocast(dtype=torch.bfloat16, device_type="cuda", enabled=True):
            for i, layer in enumerate(self.layers):
                x = layer(x, context, timestep, dropout_denoiser=dropout_denoiser, training=training)
            x = self.final_norm(x)
            if Verbose:
                print(f"[DEBUG - model] After DenoiserTower norm (before dtype clamp): dtype = {x.dtype}")
            x = x.to(torch.bfloat16)
            if Verbose:
                print(f"[DEBUG - model] After DenoiserTower norm: dtype = {x.dtype}")

        return x

## ------------------------------------------------------------
## LexaLCM Model's Main Architecture
## ------------------------------------------------------------

class LexaLCM(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.TimestepEmbedder = TimestepEmbedder(t_emb_dim=config.AdaLN_Timestep_Embed_Dim, d_model=config.d_model)

        # Architecture

        self.PreNet_C_Up = PreNetC(config.input_dim, config.d_model)

        self.ContextualizerTower = ContextualizerTower(
            num_layers=config.num_context_layers,
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            dropout=config.dropout_context
        )

        self.PostNet_C_Down = PostNetC(config.d_model, config.d_latent)

        self.LatentBridge = LatentBridge(config.d_latent, dropout=config.dropout_latent)

        self.PreNet_D_Up = PreNetD(config.d_latent, config.d_model)

        self.DenoiserTower = DenoiserTower(
            num_layers=config.num_denoiser_layers,
            d_model=config.d_model,
            d_ff=config.d_ff,
            n_heads=config.n_heads,
            dropout=config.dropout_denoiser
        )

        self.PostNet_D_Down = PostNetD(config.d_model, config.input_dim)

    def run_denoising_loop(self, latent_noise, context, *, training=False, dropout_denoiser=0.0):
        """
        Perform iterative denoising using the DenoiserTower.
        
        Args:
            latent_noise: [B, T, D] initial noisy latent input
            context: [B, T, D] fixed contextualizer output
            training: bool, whether we're training (affects # iterations and dropout)
            dropout_denoiser: float, optional dropout rate used during classifier-free guidance training

        Returns:
            denoised_latents: [B, T, D]
        """
        x = latent_noise
        num_steps = (
            self.config.denoiser_iterations_pretrain if training 
            else self.config.denoiser_iterations_inference
        )

        for t in range(num_steps):
            timestep = torch.full(
                (x.shape[0], 1, 1),
                fill_value=t,
                dtype=torch.float32,
                device=x.device
            )
            t_emb = self.TimestepEmbedder(timestep).to(x.dtype)  # [B, d_model], demote back to model dtype

            x = self.DenoiserTower(
                x,
                context,
                t_emb,
                dropout_denoiser=dropout_denoiser,
                training=training
            )

            if Verbose:
                print(f"[DEBUG - model] Denoising Loop #{t}")

        return x

    def forward(self, embeddings, labels=None):
        if Verbose:
            print(f"[DEBUG - model] embeddings: shape={embeddings.shape}, dtype={embeddings.dtype}")

        # PreNet - Contextualizer Tower

        x = self.PreNet_C_Up(embeddings)
        if Verbose:
            print(f"[DEBUG - model] after PreNet_C_Up: shape={x.shape}, dtype={x.dtype}")

        # Contextualizer Tower

        x = self.ContextualizerTower(x)
        if Verbose:
            print(f"[DEBUG - model] after ContextualizerTower: shape={x.shape}, dtype={x.dtype}")

        # PostNet - Contextualizer Tower

        x = self.PostNet_C_Down(x)
        if Verbose:
            print(f"[DEBUG - model] after PostNet_C_Down: shape={x.shape}, dtype={x.dtype}")

        # LatentBridge

        x = self.LatentBridge(x)
        if Verbose:
            print(f"[DEBUG - model] after LatentBridge: shape={x.shape}, dtype={x.dtype}")

        # PreNet - DenoiserTower

        x = self.PreNet_D_Up(x)
        if Verbose:
            print(f"[DEBUG - model] after PreNet_D_Up: shape={x.shape}, dtype={x.dtype}")

        # Denoising Loop

        # 1. Project to denoiser input space (i.e., noise dimension)
        latent_context = x  # [B, T, D]

        # 2. Create Gaussian noise matching shape of latent_context
        latent_noise = torch.randn_like(latent_context)

        # 3. Denoise from noise → latents using contextual embedding
        x = self.run_denoising_loop(
            latent_noise,
            context=latent_context,
            training=self.training,
            dropout_denoiser=self.config.dropout_denoiser
        )

        if Verbose:
            print(f"[DEBUG - model] after Denoising Loop: shape={x.shape}, dtype={x.dtype}")

        # PostNet - DenoiserTower

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