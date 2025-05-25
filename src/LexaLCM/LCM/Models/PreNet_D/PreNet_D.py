# LexaLCM/LCM/Models/PreNet_D/PreNet_D.py

import torch.nn as nn

class PreNetD(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm = nn.LayerNorm(config.in_dim)
        self.proj = nn.Linear(config.in_dim, config.out_dim)
        self.activation = nn.SiLU()

    def forward(self, x):
        x = self.norm(x)
        x = self.proj(x)
        x = self.activation(x)
        return x
