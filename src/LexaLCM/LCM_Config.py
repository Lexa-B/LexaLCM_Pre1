# src/LexaLCM/LCM_Config.py

from transformers import PretrainedConfig, CONFIG_MAPPING

class LexaLCMConfig(PretrainedConfig):
    model_type = "lexa_lcm_pre1"
    def __init__(
        self,
        input_dim=1024,
        d_model=2048,
        d_latent=1024,
        num_context_layers=5,
        num_denoiser_layers=13,
        n_heads=16,
        d_ff=8192,
        dropout_context=0.1,
        dropout_latent=0.1,
        dropout_denoiser=0.15,
        denoiser_iterations_pretrain = 100,
        denoiser_iterations_inference = 40,
        AdaLN_Timestep_Embed_Dim = 256,
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
        self.dropout_context = dropout_context
        self.dropout_latent = dropout_latent
        self.dropout_denoiser = dropout_denoiser
        self.denoiser_iterations_pretrain = denoiser_iterations_pretrain
        self.denoiser_iterations_inference = denoiser_iterations_inference
        self.AdaLN_Timestep_Embed_Dim = AdaLN_Timestep_Embed_Dim

CONFIG_MAPPING.register("lexa_lcm_pre1", LexaLCMConfig)
