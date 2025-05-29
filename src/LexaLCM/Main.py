# src/LexaLCM/Main.py

import yaml
import argparse
import wandb
import torch

from LexaLCM.LCM_Config import LexaLCMConfig
from LexaLCM.LCM_Model import LexaLCM
from LexaLCM.Data.DataHandler import LCMDataset, LCMDataset_DryRun, LCMCollator
from transformers import Trainer, TrainingArguments

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def RunTraining(config_training, model, train_dataset, val_dataset=None, dry_run=False, resume_from_checkpoint=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if not dry_run:
        model.train()
    else:
        model.eval()

    print(f"Trainable Parameters: {count_trainable_params(model):,}")

    training_args = TrainingArguments(
        output_dir=config_training['training']['output_dir'],
        per_device_train_batch_size=1 if dry_run else config_training['training']['batch_size'],
        bf16=config_training['training']['bf16'],
        max_steps=1 if dry_run else config_training['training']['max_steps'],
        logging_steps=1 if dry_run else config_training['training']['log_every'],
        eval_strategy="no" if dry_run else "steps",
        eval_steps=None if dry_run else config_training['training']['eval_every'],
        save_steps=config_training['training']['save_every'],
        learning_rate=config_training['training']['learning_rate'],
        weight_decay=config_training['training']['weight_decay'],
        warmup_steps=config_training['training']['warmup_steps'],
        max_grad_norm=config_training['training']['max_grad_norm'],
        run_name=config_training['wandb']['run_name'],
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=LCMCollator(),
    )

    print("\n🚀 Starting training...")
    trainer.train(resume_from_checkpoint=None if dry_run else resume_from_checkpoint)
    print("✅ Training complete!")
 

def LoadConfig(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def Main():
    print("🚀 Starting LexaLCM Pre1 Training")

    # CLI Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dry-run", action="store_true", help="Run a single batch through the model for sanity check.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print additional debug output.")
    args = parser.parse_args()

    # Load Config
    print("🔍 Loading config...")
    config_path = 'src/LexaLCM/Config/Pretrain/Config_Pretrain_Pre1_001.yaml'
    config_training = LoadConfig(config_path)

    if args.verbose:
        print(f"Config loaded: keys = {list(config_training.keys())}")
        print("🪵 Verbose mode ON")

    # Init WandB
    if "wandb" in config_training:
        wandb.init(
            project=config_training["wandb"]["project"],
            name=config_training["wandb"].get("run_name", None),
            config=config_training
        )

    # Init Model
    model_config = LexaLCMConfig()
    model = LexaLCM(model_config)

    # Dry Run
    if args.dry_run:
        print("🧪 Running dry run with test embeddings...")
        dataset = LCMDataset_DryRun()
        RunTraining(config_training, model, train_dataset=dataset, dry_run=True)
        return

    # Full Training
    print("📚 Loading full training datasets...")
    data_conf = config_training["data"]
    max_len = config_training['training'].get('max_seq_len', None)
    if args.verbose:
        print(f"Max sequence length: {max_len}")

    train_dataset = LCMDataset(
        data_dir=data_conf["data_dir"],
        split=data_conf["train_split"],
        text_column=data_conf["text_column"],
        max_seq_len=max_len
    )
    val_dataset = LCMDataset(
        data_dir=data_conf["data_dir"],
        split=data_conf["val_split"],
        text_column=data_conf["text_column"],
        max_seq_len=max_len
    )

    RunTraining(config_training, model, train_dataset, val_dataset, dry_run=False)

if __name__ == "__main__":
    Main()
