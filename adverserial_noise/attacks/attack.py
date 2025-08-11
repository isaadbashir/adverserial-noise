"""
Adversarial attack framework with unified interface for multiple backends.

This module provides a high-level interface for running adversarial attacks on
deep learning models. It supports multiple backends (PyTorch, TensorFlow) and
attack types (FGSM, etc.) with automatic backend detection and minimal configuration.

The module includes:
- BackendTypes: Enumeration of supported backend frameworks
- AttackTypes: Enumeration of supported attack algorithms
- AdversarialAttack: Main interface class for running attacks

Example:
    >>> # Initialize with auto-detection
    >>> attack_interface = AdversarialAttack()
    >>>
    >>> # Run FGSM attack
    >>> adversarial_inputs = attack_interface.run_attack(
    >>>     model=pretrained_model,
    >>>     inputs=input_tensor,
    >>>     targets=target_tensor,
    >>>     attack_type="fgsm",
    >>>     epsilon=0.03
    >>> )
    >>>
    >>> # Initialize with specific backend
    >>> pytorch_attack = AdversarialAttack(backend="pytorch")
"""

from typing import Optional, Union, Any, Callable
import torch
from dataclasses import dataclass

from .base.pytorch_interface import PyTorchBackend
from .fgsma_attack import FGSMAttack
from .pgd_attack import PGDAttack
from attack_interface import AdversarialAttackBase


@dataclass(frozen=True)
class BackendTypes:
    """
    Enumeration of supported backend frameworks for adversarial attacks.

    This class defines the supported deep learning frameworks that can be used
    with the adversarial attack system. It provides a centralized way to
    reference backend types and validate backend selections.

    Attributes:
        PYTORCH (str): PyTorch framework identifier.
        TENSORFLOW (str): TensorFlow framework identifier.

    Example:
        >>> # Check if a backend is supported
        >>> backend = "pytorch"
        >>> if backend in BackendTypes.values():
        >>>     print(f"Backend {backend} is supported")
        >>>
        >>> # Get all supported backends
        >>> supported = BackendTypes.values()
        >>> print(f"Supported backends: {supported}")
    """

    PYTORCH: str = "pytorch"
    TENSORFLOW: str = "tensorflow"

    @classmethod
    def values(cls) -> set[str]:
        """
        Get all supported backend types as a set.

        Returns:
            set[str]: Set containing all supported backend identifiers.

        Example:
            >>> backends = BackendTypes.values()
            >>> print(backends)  # {'pytorch', 'tensorflow'}
        """
        return {cls.PYTORCH, cls.TENSORFLOW}


@dataclass(frozen=True)
class AttackTypes:
    """
    Enumeration of supported adversarial attack algorithms.

    This class defines the supported attack types that can be executed through
    the adversarial attack interface. It provides a centralized way to
    reference attack types and validate attack selections.

    Attributes:
        FGSM (str): Fast Gradient Sign Method attack identifier.
        PGD (str): Projected Gradient Descent attack identifier.

    Example:
        >>> # Check if an attack type is supported
        >>> attack = "fgsm"
        >>> if attack in AttackTypes.values():
        >>>     print(f"Attack {attack} is supported")
        >>>
        >>> # Get all supported attacks
        >>> supported = AttackTypes.values()
        >>> print(f"Supported attacks: {supported}")
    """

    FGSM: str = "fgsm"
    PGD: str = "pgd"

    @classmethod
    def values(cls) -> set[str]:
        """
        Get all supported attack types as a set.

        Returns:
            set[str]: Set containing all supported attack identifiers.

        Example:
            >>> attacks = AttackTypes.values()
            >>> print(attacks)  # {'fgsm'}
        """
        return {cls.FGSM, cls.PGD}


# Global constants for easy access
SUPPORTED_BACKENDS = BackendTypes.values()
SUPPORTED_ATTACKS = AttackTypes.values()


class AdversarialAttack:
    """
    User-friendly interface to run adversarial attacks with minimal configuration.

    This class provides a high-level interface for executing adversarial attacks
    on deep learning models. It automatically detects the backend framework
    from the model type and provides a unified interface regardless of the
    underlying framework (PyTorch, TensorFlow).

    The interface supports multiple attack types and automatically handles
    backend-specific operations, making it easy to run attacks without
    worrying about framework-specific implementation details.

    Key Features:
        - Automatic backend detection from model type
        - Support for multiple attack algorithms
        - Unified interface across different frameworks
        - Minimal configuration requirements
        - Verbose logging options

    Example:
        >>> # Create interface with auto-detection
        >>> attack_interface = AdversarialAttack()
        >>>
        >>> # Run FGSM attack
        >>> adversarial_inputs = attack_interface.run_attack(
        >>>     model=resnet_model,
        >>>     inputs=image_batch,
        >>>     targets=labels,
        >>>     attack_type="fgsm",
        >>>     epsilon=0.03,
        >>>     verbose=True
        >>> )
        >>>
        >>> # Create interface with specific backend
        >>> pytorch_attack = AdversarialAttack(backend="pytorch")
    """

    def __init__(self, backend: Optional[str] = None) -> None:
        """
        Initialize the adversarial attack interface.

        Args:
            backend (str, optional): Backend framework to use. Can be:
                - "pytorch": Force PyTorch backend
                - "tensorflow": Force TensorFlow backend
                - None: Auto-detect backend from model type (default)

        Raises:
            ValueError: If the specified backend is not supported.

        Example:
            >>> # Auto-detect backend
            >>> interface = AdversarialAttack()
            >>>
            >>> # Force PyTorch backend
            >>> pytorch_interface = AdversarialAttack(backend="pytorch")
            >>>
            >>> # Force TensorFlow backend
            >>> tf_interface = AdversarialAttack(backend="tensorflow")

        Note:
            When backend is None, the system will automatically detect the
            backend from the model type when run_attack() is called. This
            is the recommended approach for most use cases.
        """
        if backend is not None and backend.lower() not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported backend '{backend}'. Supported: {SUPPORTED_BACKENDS}"
            )
        self.backend_name = backend.lower() if backend else None

    def _detect_backend(self, model: Any) -> str:
        """
        Automatically detect the backend framework from the model type.

        This method analyzes the model object to determine which backend
        framework it belongs to. It's used when no explicit backend is
        specified during initialization.

        Args:
            model (Any): The model object to analyze for backend detection.

        Returns:
            str: Detected backend name (e.g., "pytorch", "tensorflow").

        Raises:
            ValueError: If the backend cannot be automatically detected.

        Example:
            >>> interface = AdversarialAttack()
            >>> backend = interface._detect_backend(torch.nn.Linear(10, 5))
            >>> print(backend)  # "pytorch"

        Note:
            - PyTorch models are detected by checking if they inherit from torch.nn.Module
            - TensorFlow models would need similar detection logic
            - If detection fails, users should explicitly specify the backend
        """
        if self.backend_name:
            return self.backend_name
        if isinstance(model, torch.nn.Module):
            return BackendTypes.PYTORCH
        else:
            raise ValueError(
                "Cannot detect backend from model type. Please specify 'backend' explicitly."
            )

    def _create_backend(self, backend_name: str) -> Union[PyTorchBackend]:
        """
        Create a backend instance for the specified framework.

        This method instantiates the appropriate backend class based on the
        backend name. Each backend handles framework-specific operations
        like tensor conversion and device management.

        Args:
            backend_name (str): Name of the backend to create.

        Returns:
            Union[PyTorchBackend]: Backend instance for the specified framework.

        Raises:
            ValueError: If the backend name is unknown or not supported.

        Example:
            >>> interface = AdversarialAttack()
            >>> backend = interface._create_backend("pytorch")
            >>> print(type(backend))  # <class 'PyTorchBackend'>

        Note:
            Currently only PyTorch backend is fully implemented.
            TensorFlow backend support is planned for future versions.
        """
        if backend_name == BackendTypes.PYTORCH:
            return PyTorchBackend()
        else:
            raise ValueError(f"Unknown backend '{backend_name}'")

    def run_attack(
        self,
        model: Any,
        inputs: Any,
        targets: Any,
        attack_type: str,
        epsilon: float = 0.03,
        loss_fn: Optional[Callable[..., Any]] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Run the specified adversarial attack on inputs.

        This method executes the selected adversarial attack algorithm on the
        provided inputs using the specified model. It automatically handles
        backend detection, attack instantiation, and execution.

        The method supports various attack parameters and provides options for
        custom loss functions and verbose logging. It's designed to be the
        main entry point for running adversarial attacks.

        Args:
            model (Any): Pretrained model to attack. Can be PyTorch or TensorFlow model.
                The model should accept inputs and return logits or probabilities.
            inputs (Any): Input data to generate adversarial examples for.
                Should be compatible with the model's expected input format.
            targets (Any): Target labels or ground truth for the inputs.
                Used by the attack algorithm to compute the loss.
            attack_type (str, optional): Type of adversarial attack to run.
                Must be one of the supported attack types. Defaults to "fgsm".
            epsilon (float, optional): Maximum perturbation magnitude allowed.
                Controls how much the adversarial example can differ from the original.
                Defaults to 0.03.
            loss_fn (Callable[..., Any], optional): Custom loss function to use.
                If None, the attack will use a default loss function appropriate
                for the attack type. Defaults to None.
            verbose (bool, optional): Enable verbose logging during attack execution.
                Provides detailed information about attack progress and parameters.
                Defaults to False.
            **kwargs: Additional attack-specific parameters:
                - For PGD attacks: step_size, num_iterations, random_start
                - For FGSM attacks: no additional parameters needed

        Returns:
            Any: Adversarial examples with the same shape and type as the inputs.
                The adversarial examples are perturbed versions of the original
                inputs that are designed to fool the model.

        Raises:
            ValueError: If the attack type is not supported or backend detection fails.
            RuntimeError: If the attack execution fails or model inference errors occur.

        Example:
            >>> # Basic FGSM attack
            >>> interface = AdversarialAttack()
            >>> adversarial_inputs = interface.run_attack(
            >>>     model=resnet_model,
            >>>     inputs=image_batch,
            >>>     targets=labels,
            >>>     epsilon=0.05
            >>> )
            >>>
            >>> # PGD attack with custom parameters
            >>> adversarial_inputs = interface.run_attack(
            >>>     model=model,
            >>>     inputs=inputs,
            >>>     targets=targets,
            >>>     attack_type="pgd",
            >>>     epsilon=0.1,
            >>>     step_size=0.01,
            >>>     num_iterations=40,
            >>>     random_start=True
            >>> )
            >>>
            >>> # Custom loss function with verbose logging
            >>> def custom_loss(logits, targets):
            >>>     return torch.nn.functional.cross_entropy(logits, targets)
            >>>
            >>> adversarial_inputs = interface.run_attack(
            >>>     model=model,
            >>>     inputs=inputs,
            >>>     targets=targets,
            >>>     attack_type="fgsm",
            >>>     epsilon=0.1,
            >>>     loss_fn=custom_loss,
            >>>     verbose=True
            >>> )

        Note:
            - The method automatically detects the backend from the model type
            - All supported attack types use the same interface
            - The epsilon parameter controls the trade-off between attack strength
              and perturbation visibility
            - Custom loss functions should accept model outputs and targets
            - Verbose logging is useful for debugging and understanding attack behavior
        """
        backend_name = self._detect_backend(model)
        backend = self._create_backend(backend_name)

        attack: AdversarialAttackBase = None

        attack_type = attack_type.lower()
        if attack_type not in SUPPORTED_ATTACKS:
            raise ValueError(
                f"Unsupported attack '{attack_type}'. Choose from {SUPPORTED_ATTACKS}"
            )

        if attack_type == AttackTypes.FGSM:
            attack = FGSMAttack(backend, epsilon, loss_fn=loss_fn, verbose=verbose)
        elif attack_type == AttackTypes.PGD:
            # Extract PGD-specific parameters from kwargs with defaults
            step_size = kwargs.get("step_size", 0.01)
            num_iterations = kwargs.get("num_iterations", 40)
            random_start = kwargs.get("random_start", True)

            attack = PGDAttack(
                backend,
                epsilon,
                step_size=step_size,
                num_iterations=num_iterations,
                loss_fn=loss_fn,
                verbose=verbose,
                random_start=random_start,
            )
        else:
            raise ValueError(f"Unsupported attack type: {attack_type}")

        return attack.generate(model, inputs, targets)
