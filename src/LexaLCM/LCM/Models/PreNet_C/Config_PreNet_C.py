# LexaLCM/LCM/Models/PreNet_C/Config_PreNet_C.py

from transformers import PretrainedConfig

class PreNetCConfig(PretrainedConfig):
    model_type = "PreNetC"

    def __init__(self, in_dim=1024, out_dim=2048, **kwargs):
        super().__init__(**kwargs)
        self.in_dim = in_dim
        self.out_dim = out_dim