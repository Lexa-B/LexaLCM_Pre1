# LexaLCM/LCM/Models/PreNet_D/PreNet_D.py

import torch.nn as nn

class PreNetD(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, x):
        return x
