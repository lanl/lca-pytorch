import torch


Tensor = torch.Tensor


def make_zero_mean(batch: Tensor) -> Tensor:
    """Make each sample in a batch have zero mean"""
    dims = tuple(range(1, len(batch.shape)))
    mean = batch.mean(dim=dims, keepdim=True)
    return batch - mean


def make_unit_var(batch: Tensor, eps: float = 1e-12) -> Tensor:
    """Make each sample in a batch have unit variance"""
    dims = tuple(range(1, len(batch.shape)))
    std = batch.std(dim=dims, keepdim=True)
    return batch / (std + eps)
