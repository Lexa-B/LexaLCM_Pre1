# LexaLCM/LCM/Models/PostNet_C/Config_PostNet_C.py

from transformers import PretrainedConfig

class PostNetCConfig(PretrainedConfig):
    model_type = "PostNetC"

    def __init__(self, in_dim=2048, out_dim=1024, **kwargs):
        super().__init__(**kwargs)
        self.in_dim = in_dim
        self.out_dim = out_dim