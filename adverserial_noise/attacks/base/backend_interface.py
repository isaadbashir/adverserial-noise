from abc import ABC, abstractmethod
from typing import Any, Union
import numpy as np


class BackendInterface(ABC):
    """Defines backend-agnostic tensor and model operations."""

    @abstractmethod
    def to_tensor(
        self, data: Union[np.ndarray[Any, Any], list[Any], tuple[Any, ...], float, int]
    ) -> Any:
        pass

    @abstractmethod
    def requires_grad(self, tensor: Any) -> Any:
        pass

    @abstractmethod
    def zero_grad(self, model: Any) -> None:
        pass

    @abstractmethod
    def backward(self, loss: Any) -> None:
        pass

    @abstractmethod
    def clamp(self, tensor: Any, min_val: float, max_val: float) -> Any:
        pass

    @abstractmethod
    def device(self, tensor: Any) -> Any:
        pass

    @abstractmethod
    def model_eval(self, model: Any) -> None:
        pass

    @abstractmethod
    def gather(self, outputs: Any, targets: Any) -> Any:
        pass

    @abstractmethod
    def grad_sign(self, tensor: Any) -> Any:
        """
        Returns the element-wise sign of the gradient tensor.
        """
        pass
