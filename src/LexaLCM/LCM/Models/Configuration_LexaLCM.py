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
        input_dim=1024,          # SONAR embedding dim
        hidden_dim=2048,         # internal model dim, shared between all layers in both towers
        **kwargs
    ):
        super().__init__(**kwargs)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.contextualizer_config = ContextualizerConfig(hidden_size=hidden_dim)
        self.denoiser_config = DenoiserConfig(hidden_size=hidden_dim)
        self.prenet_c_config = PreNetCConfig(in_dim=input_dim, out_dim=hidden_dim)
        self.prenet_d_config = PreNetDConfig(in_dim=input_dim, out_dim=hidden_dim)
        self.postnet_c_config = PostNetCConfig(in_dim=hidden_dim, out_dim=input_dim)
        self.postnet_d_config = PostNetDConfig(in_dim=hidden_dim, out_dim=input_dim)
