# Tests/TestInference.py

import torch
from LexaLCM.LCM_Model import LexaLCM
from LexaLCM.LCM_Config import LexaLCMConfig
import os, sys
from torch.amp import autocast

# Add project root and SONAR path to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, project_root)

# Load SONAR Pipelines safely (calls subprocess in its own venv)
from Submodules.Pipeline_SONAR.src.pipelines import TextToEmbeddingPipeline, EmbeddingToTextPipeline

# --- Config ---
checkpoint_path = "outputs/checkpoint-1500"
test_text = "日本の生活は面白いです。"
language_code = "jpn_Jpan"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Encode input via wrapper ---
encoder = TextToEmbeddingPipeline(language=language_code, verbose=True)
embedding = encoder(test_text)   # shape: [D]
embedding = embedding.unsqueeze(0).to(device)  # [1, D]
print(f"→ Encoded: {embedding.shape}, dtype: {embedding.dtype}")

# --- Prepare model input ---
attention_mask = torch.ones((1, 1), dtype=torch.bool, device=device)
labels = embedding.clone()

# --- Load model ---
model = LexaLCM.from_pretrained(checkpoint_path)
model.eval().to(device)

# --- Inference ---
with torch.no_grad(), autocast(dtype=torch.bfloat16, device_type="cuda"):
    output = model(
        embeddings=embedding.unsqueeze(1),
        labels=labels,
        attention_mask=attention_mask
    )

pred = output["logits"]  # [B, D]
print(f"→ Output shape: {pred.shape}, dtype: {pred.dtype}")
print(f"→ Loss: {output['loss'].item():.4f}")

# --- Decode output via wrapper ---
decoder = EmbeddingToTextPipeline(language=language_code, verbose=True)
decoded = decoder(pred.squeeze(1))
print(f"→ Decoded text: {decoded}")
