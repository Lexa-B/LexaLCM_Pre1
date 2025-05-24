# Data/Collate.py

import torch
from torch.nn.utils.rnn import pad_sequence

def CollateSONARBatch(batch):
    """
    Pads a batch of [seq_len, emb_dim] tensors to the longest sequence.
    Returns: padded_batch [B, max_seq_len, emb_dim], attention_mask [B, max_seq_len]
    """
    # Pad each example to the max seq_len in the batch
    padded_batch = pad_sequence(batch, batch_first=True)  # [B, S, D]

    # Build attention mask: 1 for real tokens, 0 for padding
    attention_mask = torch.zeros(padded_batch.shape[:2], dtype=torch.bool)
    for i, seq in enumerate(batch):
        attention_mask[i, :seq.shape[0]] = 1

    return padded_batch, attention_mask
