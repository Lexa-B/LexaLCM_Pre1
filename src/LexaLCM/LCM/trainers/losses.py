import torch
import torch.nn.functional as F

def compute_loss(pred, target, mask=None):
    """
    Computes MSE loss between predicted and target embeddings, optionally masked.
    """
    if mask is not None:
        # Only compute MSE over non-padded elements
        mask = mask.unsqueeze(-1).expand_as(pred)  # [B, S, D]
        pred = pred[mask]
        target = target[mask]
    return F.mse_loss(pred, target)
