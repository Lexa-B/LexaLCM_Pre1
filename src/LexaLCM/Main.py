# src/LexaLCM/Main.py

from LexaLCM.LCM_Config import LexaLCMConfig
from LexaLCM.LCM_Model import LexaLCM
from LexaLCM.Data.DataHandler import LCMDataset, LCMCollator
from transformers import Trainer, TrainingArguments


def test_data_pipeline():
    # Test dataset and collator
    dataset = LCMDataset()
    collator = LCMCollator()
    batch = collator([dataset[0]])
    print("Test batch shapes:")

    print(f"Embeddings: {batch['embeddings'].shape}")  # [1, 3, 1024]
    print(f"Labels: {batch['labels'].shape}")  # [1, 1024]

def test_training_step():
    # Mini-training loop
    config = LexaLCMConfig()
    model = LexaLCM(config)
    training_args = TrainingArguments(
        output_dir="./tmp_output",
        per_device_train_batch_size=1,
        bf16=True,
        max_steps=1,
        logging_steps=1,
        remove_unused_columns=False
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=LCMDataset(),
        data_collator=LCMCollator(),
    )
    print("\nStarting test training...")
    trainer.train()
    print("Test training completed!")

if __name__ == "__main__":
    test_data_pipeline()
    test_training_step()

    # # Test forward pass
    # sample = SonarDataset()[0]
    # output = model(sample["embeddings"], sample["labels"])
    # print(f"Loss: {output['loss'].item()}")  # Should be ~0.5-2.0 initially
    # print(f"Output shape: {output['logits'].shape}")  # (1, 1024)

