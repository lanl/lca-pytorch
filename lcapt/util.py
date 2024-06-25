import torch


Tensor = torch.Tensor


def check_equal_shapes(tensor1: Tensor, tensor2: Tensor) -> None:
    """Checks to see if tensor1 and tensor2 have the same shape."""
    if tensor1.shape != tensor2.shape:
        raise RuntimeError(
            f"Expected shape {tensor1.shape}, but got {tensor2.shape}."
        )


def check_equal_dtypes(tensor1: Tensor, tensor2: Tensor) -> None:
    """Checks to see if tensor1 and tensor2 have the same dtype."""
    if tensor1.dtype != tensor2.dtype:
        raise RuntimeError(
            f"Expected dtype {tensor1.dtype}, but got {tensor2.dtype}."
        )


def to_5d_from_3d(inputs: Tensor) -> Tensor:
    assert len(inputs.shape) == 3
    return inputs.unsqueeze(-1).unsqueeze(-1)


def to_5d_from_4d(inputs: Tensor) -> Tensor:
    assert len(inputs.shape) == 4
    return inputs.unsqueeze(-3)


def to_3d_from_5d(inputs: Tensor) -> Tensor:
    assert len(inputs.shape) == 5
    assert inputs.shape[-2] == 1 and inputs.shape[-1] == 1
    return inputs[..., 0, 0]


def to_4d_from_5d(inputs: Tensor) -> Tensor:
    assert len(inputs.shape) == 5
    assert inputs.shape[-3] == 1
    return inputs[..., 0, :, :]
