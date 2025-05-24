# LCM/Main.py

import yaml
import torch
import wandb
import argparse

from LexaLCM.LCM.Data.Dataset import SonarDataset
from LexaLCM.LCM.Data.Collate import CollateSONARBatch
from LexaLCM.LCM.Models.Architecture_LCM import TwoTowerLCM
from LexaLCM.LCM.Trainers.Trainer import Train, LoadCheckpoint 
from LexaLCM.LCM.Trainers.Losses import ComputeLoss

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dry-run", action="store_true", help="Run a single batch through the model for sanity check.")
parser.add_argument("-v", "--verbose", action="store_true", help="Print additional debug output.")
args = parser.parse_args()

def LoadConfig(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def Main():
    print("🚀 Starting LexaLCM Pre1 Training")
    config = LoadConfig('src/LexaLCM/LCM/Configs/Config-Pretrain.yaml')

    # Initialize wandb
    if "wandb" in config:
        wandb.init(
            project=config["wandb"]["project"],
            name=config["wandb"].get("run_name", None),
            config=config
        )

    print("🔍 Loading config...")
    if args.verbose:    
        print(f"Config loaded: keys = {list(config.keys())}")
        print(f"Loading datasets...")

    # Load datasets
    data_dir = config['data']['data_dir']
    text_column = config['data']['text_column']
    train_split = config['data']['train_split']
    val_split = config['data']['val_split']

    train_dataset = SonarDataset(data_dir, train_split, text_column)
    val_dataset   = SonarDataset(data_dir, val_split, text_column)

    if args.verbose:
        print("🪵 Verbose mode ON")
        print(f"Config loaded: keys = {list(config.keys())}")
        print(f"Train dataset length: {len(train_dataset)}")
        print(f"Val dataset length: {len(val_dataset)}")

        try:
            print("⏳ Checking first train item shape...")
            example = train_dataset[0]
            if isinstance(example, torch.Tensor):
                print(f"✅ First example shape: {example.shape}")
            else:
                print(f"✅ First example type: {type(example)}, value shape: {example[0].shape}")
        except Exception as e:
            print(f"❌ Failed to access dataset[0]: {e}")

    if args.dry_run:
        print("🔍 Dry run mode: starting sanity check...")

        print("⚙️ Loading config and initializing model...")
        model = TwoTowerLCM(config['model'])

        print("📦 Loading batch...")
        loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, collate_fn=CollateSONARBatch)

        try:
            batch, mask = next(iter(loader))
        except Exception as e:
            print("❌ Failed to get batch from DataLoader:", e)
            return

        print(f"✅ Got batch: {batch.shape}, Mask: {mask.shape}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch, mask = batch.to(device), mask.to(device)
        model.to(device)
        model.eval()

        print("🚀 Running forward pass...")
        with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            output = model(batch, attention_mask=mask)

        loss = ComputeLoss(output, batch, mask=mask)

        print(f"✅ Dry run complete.\nOutput shape: {output.shape}\nLoss: {loss.item():.4f}")
        return
    else:
        print("🚦 Launching full training...")

        # Initialize model
        model = TwoTowerLCM(config['model'])
        model.cuda()
        model.eval()

        # Optimizer must be created before loading checkpoint
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )

        # Optional: Resume from checkpoint
        resume_from = config['training'].get('resume_from', None)
        start_step = 0
        if resume_from:
            print(f"Resuming from checkpoint: {resume_from}")
            start_step = LoadCheckpoint(model, optimizer, resume_from)

        # Train!
        Train(model, train_dataset, val_dataset, config, optimizer, start_step=start_step)

if __name__ == "__main__":
    Main()
