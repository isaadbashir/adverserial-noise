from abc import ABC, abstractmethod
from typing import Any, Union
import numpy as np


class BackendInterface(ABC):
    """
    Abstract base class defining backend-agnostic tensor and model operations.

    This interface provides a common contract that all framework backends must
    implement. It abstracts away framework-specific details, allowing attack
    implementations to work with any supported deep learning framework.

    The interface covers essential operations needed for adversarial attacks:
    - Tensor creation and manipulation
    - Gradient computation and handling
    - Model state management
    - Device and memory operations

    Implementations should provide efficient, framework-optimized versions of
    these operations while maintaining the same interface contract.

    Example:
        >>> class PyTorchBackend(BackendInterface):
        >>>     def to_tensor(self, data):
        >>>         return torch.tensor(data)
        >>>
        >>> backend = PyTorchBackend()
        >>> tensor = backend.to_tensor([1, 2, 3])

    Note:
        All methods in this interface are abstract and must be implemented
        by concrete backend classes. The implementations should handle
        framework-specific optimizations and error cases appropriately.
    """

    @abstractmethod
    def to_tensor(
        self, data: Union[np.ndarray[Any, Any], list[Any], tuple[Any, ...], float, int]
    ) -> Any:
        """
        Convert various data types to framework-specific tensors.

        This method should handle conversion from common Python data types
        to the framework's tensor format. It's used for creating tensors
        from numpy arrays, Python lists, or scalar values.

        Args:
            data: Input data to convert. Can be:
                - numpy.ndarray: Multi-dimensional array
                - list: Python list of values
                - tuple: Python tuple of values
                - float: Scalar float value
                - int: Scalar integer value

        Returns:
            Framework-specific tensor with the same data and shape as input.

        Note:
            The implementation should handle data type inference and
            maintain the original data precision when possible.
        """
        pass

    @abstractmethod
    def requires_grad(self, tensor: Any) -> Any:
        """
        Enable gradient computation for a tensor.

        This method sets the tensor to require gradients, which is essential
        for adversarial attacks that need to compute gradients with respect
        to the input. The tensor should be modified in-place or a new tensor
        with gradients enabled should be returned.

        Args:
            tensor: Input tensor that needs gradients enabled.

        Returns:
            Tensor with gradients enabled (can be the same tensor or a new one).

        Note:
            This is a critical operation for adversarial attacks as gradients
            are needed to compute the perturbation direction.
        """
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
        Return the element-wise sign of the gradient tensor.

        This method extracts the sign of gradients from a tensor that has
        gradients computed. It's a key operation in gradient-based attacks
        like FGSM, where the sign of gradients determines the perturbation
        direction.

        Args:
            tensor: Tensor that should have gradients computed (grad is not None).

        Returns:
            Tensor with the same shape as input, containing only -1, 0, or 1
            representing the sign of each gradient element.

        Raises:
            RuntimeError: If the tensor has no gradients computed.

        Note:
            This method should be called after backward() has been called
            on a loss that depends on the input tensor. The implementation
            should check that gradients exist before computing the sign.
        """
        pass
