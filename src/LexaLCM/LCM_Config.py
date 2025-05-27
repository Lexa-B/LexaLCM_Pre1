# src/LexaLCM/LCM_Config.py

from transformers import PretrainedConfig, CONFIG_MAPPING

class LexaLCMConfig(PretrainedConfig):
    model_type = "lexa_lcm_pre1"
    def __init__(
        self,
        input_dim=1024,
        d_model=2048,
        d_latent=1024,
        num_context_layers=1,
        num_denoiser_layers=1,
        n_heads=8,
        d_ff=8192,
        dropout=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.d_model = d_model
        self.d_latent = d_latent
        self.num_context_layers = num_context_layers
        self.num_denoiser_layers = num_denoiser_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout

CONFIG_MAPPING.register("lexa_lcm_pre1", LexaLCMConfig)
