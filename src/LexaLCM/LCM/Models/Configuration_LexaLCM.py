from transformers import PretrainedConfig

class LexaLCMConfig(PretrainedConfig):
    model_type = "lexalcm"

    def __init__(self, contextualizer_config=None, denoiser_config=None, prenet_config=None, postnet_config=None, **kwargs):
        super().__init__(**kwargs)
        self.contextualizer_config = contextualizer_config or ContextualizerConfig()
        self.denoiser_config = denoiser_config or DenoiserConfig()
        self.prenet_config = prenet_config or PreNetConfig()
        self.postnet_config = postnet_config or PostNetConfig()