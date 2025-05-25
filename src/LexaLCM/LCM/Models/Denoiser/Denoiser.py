# LexaLCM/LCM/Models/Denoiser/Denoiser.py

import torch.nn as nn

import torch.nn as nn

class Denoiser(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, x, context=None, attention_mask=None, timestep=None):
        return x

