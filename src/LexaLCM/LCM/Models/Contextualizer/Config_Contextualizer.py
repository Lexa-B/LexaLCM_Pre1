# LexaLCM/LCM/Models/Contextualizer/Config_Contextualizer.py

from transformers import PretrainedConfig

class ContextualizerConfig(PretrainedConfig):
    model_type = "Contextualizer"

    def __init__(self, hidden_size=1024, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
