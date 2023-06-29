import torch


Tensor = torch.Tensor


def compute_frac_active(acts: Tensor) -> float:
    """Computes the number of active neurons relative to the total
    number of neurons"""
    return (acts != 0.0).float().mean().item()


def compute_l1_sparsity(acts: Tensor, lambda_: float) -> Tensor:
    """Compute l1 sparsity term of objective function"""
    dims = tuple(range(1, len(acts.shape)))
    return lambda_ * acts.norm(p=1, dim=dims).mean()


def compute_l2_error(inputs: Tensor, recons: Tensor) -> Tensor:
    """Compute l2 recon error term of objective function"""
    error = inputs - recons
    dims = tuple(range(1, len(error.shape)))
    return 0.5 * (error.norm(p=2, dim=dims) ** 2).mean()


def compute_times_active_by_feature(acts: Tensor) -> Tensor:
    """Computes number of active coefficients per feature"""
    dims = list(range(len(acts.shape)))
    dims.remove(1)
    times_active = (acts != 0).float().sum(dim=dims)
    return times_active.reshape((acts.shape[1],) + (1,) * len(dims))
