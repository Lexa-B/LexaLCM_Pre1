# LexaLCM/LCM/Models/Architecture_LCM.py

from transformers import PreTrainedModel
from LexaLCM.LCM.Models.Configuration_LexaLCM import LexaLCMConfig
from LexaLCM.LCM.Models.Contextualizer.Contextualizer import Contextualizer
from LexaLCM.LCM.Models.Denoiser.Denoiser import Denoiser
from LexaLCM.LCM.Models.PreNet_C.PreNet_C import PreNetC
from LexaLCM.LCM.Models.PreNet_D.PreNet_D import PreNetD
from LexaLCM.LCM.Models.PostNet_C.PostNet_C import PostNetC
from LexaLCM.LCM.Models.PostNet_D.PostNet_D import PostNetD

class LexaLCMModel(PreTrainedModel):
    config_class = LexaLCMConfig

    def __init__(self, config):
        super().__init__(config)
        self.prenet_c = PreNetC(config.prenet_c_config)
        self.contextualizer = Contextualizer(config.contextualizer_config)
        self.postnet_c = PostNetC(config.postnet_c_config)
        self.prenet_d = PreNetD(config.prenet_d_config)
        self.denoiser = Denoiser(config.denoiser_config)
        self.postnet_d = PostNetD(config.postnet_d_config)

    def forward(self, inputs, **kwargs):
        x_c = self.prenet_c(inputs)
        context = self.contextualizer(x_c)
        output_c = self.postnet_c(context)
        x_d = self.prenet_d(inputs)
        denoised = self.denoiser(x_d, context, **kwargs)
        output_d = self.postnet_d(denoised)
        return output_d




# import torch
# import torch.nn as nn

# from LexaLCM.LCM.Models.AdaLN import AdaLayerNorm

# class SwiGLU(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.silu = nn.SiLU()

#     def forward(self, x):
#         x1, x2 = x.chunk(2, dim=-1)
#         return self.silu(x1) * x2

# def RotateHalf(x):
#     x1 = x[..., ::2]
#     x2 = x[..., 1::2]
#     return torch.cat((-x2, x1), dim=-1)

# def ApplyRotaryEmb(q, k, sin, cos):
#     # q, k: [batch, n_heads, seq_len, head_dim]
#     # sin, cos: [seq_len, head_dim]
#     q_rot = (q * cos) + (RotateHalf(q) * sin)
#     k_rot = (k * cos) + (RotateHalf(k) * sin)
#     return q_rot, k_rot

# def BuildRopeCache(seq_len, head_dim, device):
#     inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
#     t = torch.arange(seq_len, device=device).float()
#     freqs = torch.einsum('i,j->ij', t, inv_freq)
#     emb = torch.cat((freqs, freqs), dim=-1)   # [seq_len, head_dim]
#     sin = emb.sin()  # [seq_len, head_dim]
#     cos = emb.cos()
#     return sin, cos

# class RoPEMultiheadAttention(nn.Module):
#     def __init__(self, embed_dim, num_heads, max_seq_len=4096):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.head_dim = embed_dim // num_heads
#         assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
#         self.q_proj = nn.Linear(embed_dim, embed_dim)
#         self.k_proj = nn.Linear(embed_dim, embed_dim)
#         self.v_proj = nn.Linear(embed_dim, embed_dim)
#         self.out_proj = nn.Linear(embed_dim, embed_dim)

#         # Cache RoPE embeddings once in FP32
#         sin, cos = BuildRopeCache(max_seq_len, self.head_dim, 'cpu')
#         self.register_buffer('sin_cached', sin, persistent=False)
#         self.register_buffer('cos_cached', cos, persistent=False)

#     def forward(self, x, mask=None):
#         batch, seq_len, _ = x.shape
#         device = x.device

#         q = self.q_proj(x)
#         k = self.k_proj(x)
#         v = self.v_proj(x)

#         q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1,2)
#         k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1,2)
#         v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1,2)

#         # Use cached RoPE embeddings (faster and memory efficient!)
#         with torch.amp.autocast(device_type='cuda', enabled=False):
#             sin = self.sin_cached[:seq_len, :].to(device=q.device, dtype=q.dtype).unsqueeze(0).unsqueeze(0)
#             cos = self.cos_cached[:seq_len, :].to(device=q.device, dtype=q.dtype).unsqueeze(0).unsqueeze(0)

#         q, k = ApplyRotaryEmb(q, k, sin, cos)

#         # Compute attention scores
#         attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

#         if mask is not None:
#             # mask: [batch, seq_len] → [batch, 1, 1, seq_len]
#             mask = mask.unsqueeze(1).unsqueeze(2)  # broadcast to [B, 1, 1, S]
#             attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

#         attn_probs = torch.softmax(attn_scores, dim=-1)
#         attn_output = torch.matmul(attn_probs, v)

#         attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
#         output = self.out_proj(attn_output)
#         return output

# class RoPETransformerEncoderLayer(nn.Module):
#     def __init__(self, config, use_adaln=False, timestep_embed_dim=None):
#         super().__init__()
#         self.use_adaln = use_adaln

#         self.self_attn = RoPEMultiheadAttention(
#             config['hidden_size'], 
#             config['num_attention_heads'],
#             config.get('max_position_embeddings', 4096)
#         )

#         if use_adaln:
#             self.norm1 = AdaLayerNorm(config['hidden_size'], timestep_embed_dim)
#             self.norm2 = AdaLayerNorm(config['hidden_size'], timestep_embed_dim)
#         else:
#             self.norm1 = nn.LayerNorm(config['hidden_size'])
#             self.norm2 = nn.LayerNorm(config['hidden_size'])

#         self.attn_dropout = nn.Dropout(config.get('dropout', 0.1))

#         self.ffn = nn.Sequential(
#             nn.Linear(config['hidden_size'], config['hidden_size'] * 8),
#             SwiGLU(),
#             nn.Dropout(config.get('dropout', 0.1)),
#             nn.Linear(config['hidden_size'] * 4, config['hidden_size']),
#             nn.Dropout(config.get('dropout', 0.1)),
#         )

#     def forward(self, x, mask=None, timestep_emb=None):
#         norm1 = self.norm1(x, timestep_emb) if self.use_adaln else self.norm1(x)
#         attn_output = self.self_attn(norm1, mask=mask)
#         x = x + self.attn_dropout(attn_output)

#         norm2 = self.norm2(x, timestep_emb) if self.use_adaln else self.norm2(x)
#         ffn_output = self.ffn(norm2)
#         x = x + ffn_output
#         return x

# class TwoTowerLCM(nn.Module):
#     def __init__(self, config):
#         super().__init__()

#         self.num_denoising_steps = config.get('num_denoising_steps', 100)
#         self.hidden_size = config['hidden_size']

#         # Input projection for SONAR input: force FP32
#         self.input_proj = nn.Linear(config['input_dim'], config['hidden_size']).to(torch.float32)

#         # Timestep MLP
#         self.timestep_mlp = nn.Sequential(
#             nn.Linear(config['hidden_size'], config['hidden_size']),
#             nn.SiLU(),
#             nn.Linear(config['hidden_size'], config['hidden_size'])
#         )
#         # Contextualizer stack
#         self.contextualizer = nn.ModuleList([
#             RoPETransformerEncoderLayer(config)
#             for _ in range(config['num_contextualizer_layers'])
#         ])

#         # Denoiser stack (with AdaLN)
#         self.denoiser = nn.ModuleList([
#             RoPETransformerEncoderLayer(
#                 config,
#                 use_adaln=True,
#                 timestep_embed_dim=config['hidden_size']
#             ) for _ in range(config['num_denoiser_layers'])
#         ])

#     def forward(self, x, attention_mask=None, timestep=None):
#         # SONAR input projection in FP32
#         with torch.amp.autocast(device_type='cuda', enabled=False):
#             x = self.input_proj(x.float())  # always do input proj in FP32

#         # # Contextualizer (in autocast/bf16 context)
#         # for layer in self.contextualizer:
#         #     x = layer(x, mask=attention_mask)

#         # timestep_emb = self.timestep_mlp(timestep) if timestep is not None else torch.zeros(x.shape[0], x.shape[-1], device=x.device)

#         # # Denoiser
#         # for t in range(self.num_denoising_steps):
#         #     timestep_tensor = torch.full((x.shape[0], 1), t, dtype=torch.float32, device=x.device)
#         #     timestep_emb = self.timestep_mlp(timestep_tensor)  # [B, H]

#         #     h = x
#         #     for layer in self.denoiser:
#         #         h = layer(h, mask=attention_mask, timestep_emb=timestep_emb)

#         #     x = h  # pass denoised output to next timestep

#         # return x

#         print(f"[Contextualizer] Starting with input shape: {x.shape}")

#         for i, layer in enumerate(self.contextualizer):
#             x = layer(x, mask=attention_mask)
#         print(f"[Contextualizer] Done. Output shape: {x.shape}")

#         print(f"[Denoiser] Running {self.num_denoising_steps} denoising steps")

#         for t in range(self.num_denoising_steps):
#             timestep_tensor = torch.full((x.shape[0], self.hidden_size), t, dtype=torch.float32, device=x.device)
#             timestep_emb = self.timestep_mlp(timestep_tensor)

#             print(f"  ⏱️ Step {t+1}/{self.num_denoising_steps}")

#             h = x
#             for j, layer in enumerate(self.denoiser):
#                 h = layer(h, mask=attention_mask, timestep_emb=timestep_emb)
#                 if j == 0:
#                     print(f"    └ Layer {j+1}: shape = {h.shape}")  # print only once per step

#             x = h

#         return x

