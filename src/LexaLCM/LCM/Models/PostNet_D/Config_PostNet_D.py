# LexaLCM/LCM/Models/PostNet_D/Config_PostNet_D.py

from transformers import PretrainedConfig

class PostNetDConfig(PretrainedConfig):
    model_type = "PostNetD"

    def __init__(self, in_dim=2048, out_dim=1024, **kwargs):
        super().__init__(**kwargs)
        self.in_dim = in_dim
        self.out_dim = out_dim