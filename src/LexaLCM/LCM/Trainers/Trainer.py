# Trainers/Trainer.py

import torch
import os
import wandb
from torch.utils.data import DataLoader
from transformers import get_scheduler

from LexaLCM.LCM.Data.Collate import CollateSONARBatch
from LexaLCM.LCM.Trainers.Losses import ComputeLoss
from LexaLCM.LCM.Trainers.Evaluate import Evaluate

def SaveCheckpoint(model, optimizer, step, path):
    os.makedirs(path, exist_ok=True)
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": step,
    }
    wandb.save(os.path.join(path, f"checkpoint_step_{step}.pt"))
    torch.save(checkpoint, os.path.join(path, f"checkpoint_step_{step}.pt"))
    print(f"Checkpoint saved at step {step}")

def LoadCheckpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    return checkpoint['step']

def Train(model, train_dataset, val_dataset, config, optimizer, start_step=0):
    max_steps = config['training']['max_steps']
    warmup_steps = config['training'].get('warmup_steps', 0)
    max_grad_norm = config['training'].get('max_grad_norm', 1.0)
    log_every = config['training'].get('log_every', 50)
    eval_every = config['training'].get('eval_every', 500)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=CollateSONARBatch
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        collate_fn=CollateSONARBatch
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.Train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps
    )

    step = 0

    while step < max_steps:
        for batch, mask in train_loader:
            if step >= max_steps:
                break

            batch = batch.to(device)
            mask = mask.to(device)

            with torch.cuda.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                output = model(batch, attention_mask=mask)
                loss = ComputeLoss(output, batch, mask=mask)

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if step % log_every == 0:
                log_data = {"train/loss": loss.item(), "step": step}
                log_data["lr"] = optimizer.param_groups[0]["lr"]
                print(f"Step {step}: Loss = {loss.item():.4f}")
                if wandb.run:
                    wandb.log(log_data, step=step)

            if step % eval_every == 0 and step > 0:
                Evaluate(model, val_loader, step)

            if step % config['training'].get('save_every', 1000) == 0 and step > 0:
                SaveCheckpoint(model, optimizer, step, config['training']['checkpoint_dir'])

            step += 1

    # Add validation and checkpointing as needed
