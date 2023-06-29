import torch
import torch.nn.functional as F


Tensor = torch.Tensor


def hard_threshold(x: Tensor, threshold: float, nonneg: bool = True) -> Tensor:
    """Hard threshold transfer function"""
    if nonneg:
        return F.threshold(x, threshold, 0.0)

    return F.threshold(x, threshold, 0.0) - F.threshold(-x, threshold, 0.0)


def soft_threshold(x: Tensor, threshold: float, nonneg: bool = True) -> Tensor:
    """Soft threshold transfer function"""
    if nonneg:
        return F.relu(x - threshold)

    return F.relu(x - threshold) - F.relu(-x - threshold)
