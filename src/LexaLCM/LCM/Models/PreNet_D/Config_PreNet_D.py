# LexaLCM/LCM/Models/PreNet_D/Config_PreNet_D.py

from transformers import PretrainedConfig

class PreNetDConfig(PretrainedConfig):
    model_type = "PreNetD"

    def __init__(self, in_dim=1024, out_dim=2048, **kwargs):
        super().__init__(**kwargs)
        self.in_dim = in_dim
        self.out_dim = out_dim