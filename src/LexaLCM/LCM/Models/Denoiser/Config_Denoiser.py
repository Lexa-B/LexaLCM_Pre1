# LexaLCM/LCM/Models/Denoiser/Config_Denoiser.py

from transformers import PretrainedConfig

class DenoiserConfig(PretrainedConfig):
    model_type = "Denoiser"

    def __init__(self, hidden_size=1024, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
