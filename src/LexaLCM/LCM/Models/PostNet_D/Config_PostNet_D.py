# LexaLCM/LCM/Models/PostNet_D/Config_PostNet_D.py

from transformers import PretrainedConfig

class PostNetDConfig(PretrainedConfig):
    model_type = "PostNetD"

    def __init__(self, hidden_size=1024, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size