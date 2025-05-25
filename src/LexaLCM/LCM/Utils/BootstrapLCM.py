import torch
from LexaLCM.LCM.Models.Configuration_LexaLCM import LexaLCMConfig
from LexaLCM.LCM.Models.Architecture_LCM import LexaLCMModel

def Main():
    # 🔧 Set up dummy config
    config = LexaLCMConfig(
        #hidden_dim=1024  # This should match SONAR embedding dim for now
    )

    # 🧠 Create dummy input (e.g., batch of 2 sentences, each with 10 tokens of 1024-dim SONAR embeddings)
    batch_size = 2
    seq_len = 10
    embed_dim = 1024
    dummy_input = torch.randn(batch_size, seq_len, embed_dim)

    # 🧢 Create dummy attention mask
    attention_mask = torch.ones(batch_size, seq_len)

    # ⏱️ Create dummy timestep tensor
    dummy_timestep = torch.zeros(batch_size, dtype=torch.float32)

    # 🧬 Initialize model with empty modules
    model = LexaLCMModel(config)

    if True: # Print the model's dimensions
        print("🔍 Shared hidden dim =", model.config.hidden_dim)
        print("→ Input dim =", model.config.input_dim)
        print(f"→ PreNetC dimensions = {model.config.prenet_c_config.in_dim} -> {model.config.prenet_c_config.out_dim}")
        print(f"→ Contextualizer dimensions = {model.config.contextualizer_config.hidden_size}")
        print(f"→ PostNetC dimensions = {model.config.postnet_c_config.in_dim} -> {model.config.postnet_c_config.out_dim}")
        print(f"→ PreNetD dimensions = {model.config.prenet_d_config.in_dim} -> {model.config.prenet_d_config.out_dim}")
        print(f"→ Denoiser dimensions = {model.config.denoiser_config.hidden_size}")
        print(f"→ PostNetD dimensions = {model.config.postnet_d_config.in_dim} -> {model.config.postnet_d_config.out_dim}")

    # 🖥️ Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_input = dummy_input.to(device)
    attention_mask = attention_mask.to(device)
    dummy_timestep = dummy_timestep.to(device)
    model.to(device)

    # 🔄 Forward pass
    model.eval()
    with torch.no_grad(), torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
        output = model(dummy_input, attention_mask=attention_mask, timestep=dummy_timestep)

    print(f"✅ Bootstrap complete. Output shape: {output.shape}")

if __name__ == "__main__":
    Main()
