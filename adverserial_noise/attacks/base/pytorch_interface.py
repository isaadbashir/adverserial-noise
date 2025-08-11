import torch
import numpy as np
from typing import Any, Union
from .backend_interface import BackendInterface


class PyTorchBackend(BackendInterface):
    """
    PyTorch implementation of the BackendInterface.

    This class provides PyTorch-specific implementations of all backend operations
    required for adversarial attacks. It handles tensor operations, gradient
    computation, model management, and device operations using PyTorch's API.

    The implementation is optimized for PyTorch models and tensors, providing
    efficient operations while maintaining the same interface contract as the
    abstract BackendInterface.

    Features:
        - Full PyTorch tensor support
        - Automatic device handling (CPU/GPU)
        - Efficient gradient operations
        - Model state management
        - Error handling with PyTorch-specific exceptions

    Example:
        >>> backend = PyTorchBackend()
        >>> tensor = backend.to_tensor([1, 2, 3])
        >>> tensor = backend.requires_grad(tensor)
        >>> device = backend.device(tensor)

    Note:
        This backend requires PyTorch to be installed. All operations are
        performed using PyTorch tensors and models.
    """

    def to_tensor(
        self, data: Union[np.ndarray[Any, Any], list[Any], tuple[Any, ...], float, int]
    ) -> torch.Tensor:
        return torch.tensor(data)

    def requires_grad(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.detach()
        tensor.requires_grad = True
        return tensor

    def zero_grad(self, model: torch.nn.Module) -> None:
        model.zero_grad()

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()

    def uniform(self, shape: tuple[int, ...], low: float, high: float) -> torch.Tensor:
        return torch.rand(shape) * (high - low) + low

    def clamp(
        self, tensor: torch.Tensor, min_val: float, max_val: float
    ) -> torch.Tensor:
        return torch.clamp(tensor, min_val, max_val)

    def device(self, tensor: torch.Tensor) -> torch.device:
        return tensor.device

    def model_eval(self, model: torch.nn.Module) -> None:
        model.eval()

    def gather(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "Loss functions should be supplied by the attack or user."
        )

    def grad_sign(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.grad is None:
            raise RuntimeError(
                "Tensor has no gradients. Make sure to call backward() first."
            )
        return torch.sign(tensor.grad)
