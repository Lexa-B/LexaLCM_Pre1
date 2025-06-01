import sys
import os
import torch
from torch.amp import autocast
from safetensors.torch import load_file

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, project_root)

from Submodules.Pipeline_SONAR.src.pipelines import TextToEmbeddingPipeline, EmbeddingToTextPipeline
from LexaLCM.LCM_Model import LexaLCM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
checkpoint_path = "/home/lexa/DevProjects/_Models/LexaLCM_Pre1/outputs/checkpoint-70000/"
model = LexaLCM.from_pretrained(checkpoint_path)
model.eval().cuda()

# Init pipelines
encoder = TextToEmbeddingPipeline(language="eng_Latn", verbose=True, dtype=torch.float32)
decoder = EmbeddingToTextPipeline(language="eng_Latn", verbose=True, dtype=torch.float32)

# Input prompt as sequence of sentences
prompt = ["[[Start of Text.]]", "The Sengoku era was a period of great conflict in Japan.", "Many clans and their samurai fought in that time.", "It was followed by a period of peace and cultural growth."]

# Encode each sentence and build autoregressive input
with torch.no_grad():
    context_embeddings = []
    for sentence in prompt:
        emb = encoder(sentence)  # shape: [1, 1024]
        emb = emb.to(torch.float32)
        context_embeddings.append(emb)

    # Stack to shape [1, T, 1024]
    context = torch.stack(context_embeddings, dim=0).unsqueeze(0).to(device)  # [1, T, 1024]

    print(f"→ Context shape: {context.shape}, dtype: {context.dtype}")

    # context_input = context[:, :-1, :]  # [1, T-1, 1024]
    # target = context[:, 1:, :]          # [1, T-1, 1024]

    # with autocast(dtype=torch.bfloat16, device_type="cuda"):
    #     pred = model(context_input)     # [1, T-1, 1024]

    # print(f"→ Output shape: {pred.shape}, dtype: {pred.dtype}")
    # print("→ Prediction vector sample:", pred.squeeze(1)[0, :10])

    # # Decode the last predicted embedding
    # decoded_Sonar = decoder(target[:, -1, :].squeeze(0))
    # decoded_LCM = decoder(pred[:, -1, :])
    # print(f"→ Decoded text - Last SONAR Embedding: {decoded_Sonar}")
    # print(f"→ Decoded text - Last LCM Embedding: {decoded_LCM}")

    # # Load EoT embedding
    # eot_path = "src/LexaLCM/Data/SpecialConcepts/EndOfText.safetensors"
    # eot_tensor = load_file(eot_path)["embedding"]  # Adjust the key if needed!
    # eot_tensor = eot_tensor.to(pred.device)  # [1, 1024]

    # # Grab the last timestep of pred and target (shape [1, 1024])
    # pred_vec = pred[:, -1, :]   # shape [1, 1024]
    # target_vec = target[:, -1, :] # shape [1, 1024]

    # # Compute L2 distances
    # def l2_dist(a, b):
    #     # both a and b are [1, 1024]
    #     return torch.norm(a - b, p=2).item()

    # l2_pred_eot = l2_dist(pred_vec, eot_tensor)
    # l2_true_eot = l2_dist(target_vec, eot_tensor)
    # l2_pred_true = l2_dist(pred_vec, target_vec)

    # print(f"→ L2 distance (Predicted vs. EoT): {l2_pred_eot:.4f}")
    # print(f"→ L2 distance (Ground Truth vs. EoT): {l2_true_eot:.4f}")
    # print(f"→ L2 distance (Predicted vs. Ground Truth): {l2_pred_true:.4f}")



    # context_input = context[:, :-1, :]  # [1, T-1, 1024]
    # target = context[:, 1:, :]          # [1, T-1, 1024]

    # with autocast(dtype=torch.bfloat16, device_type="cuda"):
    #     pred = model(context_input)     # [1, T-1, 1024]

    # print(f"→ Output shape: {pred.shape}, dtype: {pred.dtype}")

    # for t in range(pred.shape[1]):
    #     decoded_pred = decoder(pred[:, t, :])
    #     decoded_gt = decoder(target[:, t, :])
    #     print(f"Step {t}:")
    #     print(f"  → Model prediction: {decoded_pred}")
    #     print(f"  → Ground Truth:     {decoded_gt}")

    # # You can also print L2 distance at each step, if desired:
    # l2s = torch.norm(pred - target, dim=-1).squeeze(0)  # [T-1]
    # print("L2 distance at each step:", l2s.cpu().numpy())


with torch.no_grad():
    with autocast(dtype=torch.bfloat16, device_type="cuda"):
        pred = model(context)  # [1, 4, 1024]

# # Now: pred[:, -1, :] is the model's guess for the next embedding after the paragraph
# predicted_next_embedding = pred[:, -1, :]
# decoded_next_sentence = decoder(predicted_next_embedding)
# print("Model's prediction for the next sentence:", decoded_next_sentence)

# Optionally, print every prediction at each step (should match ground truth for S1, S2, S3, and then new guess for S4)
for t in range(pred.shape[1]):
    decoded = decoder(pred[:, t, :])
    print(f"Step {t} model next-token guess: {decoded}")