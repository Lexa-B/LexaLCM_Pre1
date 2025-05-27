# src/LexaLCM/LCM_Config.py

from transformers import PretrainedConfig, CONFIG_MAPPING

class LexaLCMConfig(PretrainedConfig):
    model_type = "lexa_lcm_pre1"
    def __init__(
        self,
        input_dim=1024,
        d_model=2048,
        d_latent=1024,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.d_model = d_model
        self.d_latent = d_latent

CONFIG_MAPPING.register("lexa_lcm_pre1", LexaLCMConfig)
