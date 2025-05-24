# models/adaln.py
import torch
import torch.nn as nn

class AdaLayerNorm(nn.Module):
    def __init__(self, hidden_size, timestep_embed_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.scale = nn.Linear(timestep_embed_dim, hidden_size)
        self.shift = nn.Linear(timestep_embed_dim, hidden_size)

    def forward(self, x, timestep_emb):
        # x: [B, S, D], timestep_emb: [B, D_emb]
        normed = self.norm(x)
        scale = self.scale(timestep_emb).unsqueeze(1)  # [B, 1, D]
        shift = self.shift(timestep_emb).unsqueeze(1)  # [B, 1, D]
        return normed * scale + shift
