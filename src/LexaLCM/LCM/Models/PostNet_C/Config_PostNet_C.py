# LexaLCM/LCM/Models/PostNet_C/Config_PostNet_C.py

from transformers import PretrainedConfig

class PostNetCConfig(PretrainedConfig):
    model_type = "PostNetC"

    def __init__(self, hidden_size=1024, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size