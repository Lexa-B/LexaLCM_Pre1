# LexaLCM/LCM/Models/Configuration_LexaLCM.py

from transformers import PretrainedConfig
from LexaLCM.LCM.Models.Contextualizer.Config_Contextualizer import ContextualizerConfig
from LexaLCM.LCM.Models.Denoiser.Config_Denoiser import DenoiserConfig
from LexaLCM.LCM.Models.PreNet_C.Config_PreNet_C import PreNetCConfig
from LexaLCM.LCM.Models.PreNet_D.Config_PreNet_D import PreNetDConfig
from LexaLCM.LCM.Models.PostNet_C.Config_PostNet_C import PostNetCConfig
from LexaLCM.LCM.Models.PostNet_D.Config_PostNet_D import PostNetDConfig

class LexaLCMConfig(PretrainedConfig):
    model_type = "lexalcm"

    def __init__(
        self,
        contextualizer_config=None,
        denoiser_config=None,
        prenet_c_config=None,
        prenet_d_config=None,
        postnet_c_config=None,
        postnet_d_config=None,
        shared_hidden_dim=1024,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.contextualizer_config = contextualizer_config or ContextualizerConfig()
        self.denoiser_config = denoiser_config or DenoiserConfig()
        self.prenet_c_config = prenet_c_config or PreNetCConfig()
        self.prenet_d_config = prenet_d_config or PreNetDConfig()
        self.postnet_c_config = postnet_c_config or PostNetCConfig()
        self.postnet_d_config = postnet_d_config or PostNetDConfig()
        self.shared_hidden_dim = shared_hidden_dim
