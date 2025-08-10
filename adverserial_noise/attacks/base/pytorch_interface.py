import torch
import numpy as np
from typing import Any, Union
from .backend_interface import BackendInterface


class PyTorchBackend(BackendInterface):
    def to_tensor(
        self, data: Union[np.ndarray[Any, Any], list[Any], tuple[Any, ...], float, int]
    ) -> torch.Tensor:
        return torch.tensor(data)

    def requires_grad(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor.requires_grad = True
        return tensor

    def zero_grad(self, model: torch.nn.Module) -> None:
        model.zero_grad()

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()

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
