# LexaLCM/LCM/Models/PostNet_C/PostNet_C.py

import torch.nn as nn

class PostNetC(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Linear(config.in_dim, config.out_dim)
        self.norm = nn.LayerNorm(config.out_dim)  # normalization after projection

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x
