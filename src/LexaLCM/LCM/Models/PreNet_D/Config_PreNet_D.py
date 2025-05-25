# LexaLCM/LCM/Models/PreNet_D/Config_PreNet_D.py

from transformers import PretrainedConfig

class PreNetDConfig(PretrainedConfig):
    model_type = "PreNetD"

    def __init__(self, hidden_size=1024, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size