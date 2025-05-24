# Trainers/Evaluate.py

import torch
import wandb

from LexaLCM.LCM.Trainers.Losses import ComputeLoss

@torch.no_grad()
def Evaluate(model, val_loader, step):
    model.eval()
    total_loss = 0.0
    total_batches = 0

    device = next(model.parameters()).device

    for batch, mask in val_loader:
        batch = batch.to(device)
        mask = mask.to(device)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            output = model(batch, attention_mask=mask)
            loss = ComputeLoss(output, batch, mask=mask)

        total_loss += loss.item()
        total_batches += 1

    avg_loss = total_loss / max(1, total_batches)
    
    print(f"Validation Loss: {avg_loss:.4f}")

    if wandb.run:
        wandb.log({"val/loss": avg_loss, "step": step})

    model.Train()
