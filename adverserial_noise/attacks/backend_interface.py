"""Backend interface for PyTorch and TensorFlow operations."""

import torch
import tensorflow as tf
from typing import Union, Any


class PyTorchBackend:
    """PyTorch-specific backend operations."""

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def to_tensor(self, data: Union[torch.Tensor, Any]) -> torch.Tensor:
        if not isinstance(data, torch.Tensor):
            return torch.tensor(data, device=self.device)
        return data.to(self.device)


class TensorFlowBackend:
    """TensorFlow-specific backend operations."""

    def __init__(self) -> None:
        pass

    def to_tensor(self, data: Union[tf.Tensor, Any]) -> tf.Tensor:
        if not isinstance(data, tf.Tensor):
            return tf.convert_to_tensor(data)
        return data
