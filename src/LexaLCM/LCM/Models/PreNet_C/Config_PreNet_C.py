# LexaLCM/LCM/Models/PreNet_C/Config_PreNet_C.py

from transformers import PretrainedConfig

class PreNetCConfig(PretrainedConfig):
    model_type = "PreNetC"

    def __init__(self, hidden_size=1024, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size