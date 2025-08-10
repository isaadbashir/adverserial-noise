from abc import ABC, abstractmethod
from typing import Optional, Callable, Any, Tuple
from .base.backend_interface import BackendInterface


class AdversarialAttackBase(ABC):
    """
    Framework-agnostic base class for adversarial attacks.

    This abstract base class provides a common interface for implementing
    adversarial attacks that can work with any deep learning framework.
    It handles common functionality like input validation, gradient computation,
    and attack orchestration while delegating framework-specific operations
    to backend implementations.

    The class is designed to be extended by concrete attack implementations
    that only need to implement the `_perturb` method with their specific
    attack logic.

    Attributes:
        backend (BackendInterface): Backend instance for tensor/model operations.
        epsilon (float): Maximum perturbation magnitude (L-infinity norm).
        targeted (bool): Whether the attack is targeted (misclassify to specific class)
                        or untargeted (misclassify to any wrong class).
        loss_fn (Callable): Loss function to optimize during the attack.
        verbose (bool): Enable verbose output for debugging and monitoring.

    Example:
        >>> class CustomAttack(AdversarialAttackBase):
        >>>     def _perturb(self, model, inputs, targets):
        >>>         # Implement custom attack logic here
        >>>         return adversarial_inputs, perturbation
        >>>
        >>> attack = CustomAttack(backend, epsilon=0.1, loss_fn=custom_loss)
        >>> adv_inputs, noise = attack.generate(model, inputs, targets)

    Note:
        All attacks should inherit from this class and implement the `_perturb`
        method. The `generate` method provides the main interface for running
        attacks and handles common validation and setup.
    """

    def __init__(
        self,
        backend: BackendInterface,
        epsilon: float,
        targeted: bool = True,
        loss_fn: Optional[Callable[[Any, Any], Any]] = None,
        verbose: bool = False,
    ):
        self.backend = backend
        self.epsilon = epsilon
        self.targeted = targeted

        if loss_fn is None:
            raise ValueError("loss_fn must be provided")
        self.loss_fn = loss_fn
        self.verbose = verbose

    def _validate_inputs(self, model: Any, inputs: Any, targets: Any) -> None:
        """
        Validate input parameters for the attack.

        This method performs basic validation of the model, inputs, and targets
        to ensure they meet the requirements for adversarial attack generation.
        It checks for None values, tensor shapes, and batch size consistency.

        Args:
            model (Any): The model to attack. Must not be None.
            inputs (Any): Input batch tensor. Must be 4D (batch, channels, height, width).
            targets (Any): Target labels. Must be 1D tensor/array of class indices.

        Raises:
            ValueError: If any validation check fails.

        Note:
            This is a basic validation implementation. Subclasses can override
            this method to add framework-specific or attack-specific validation.
        """
        if inputs is None or targets is None or model is None:
            raise ValueError("Model, inputs, and targets must not be None")

        # Input tensor shape check for 4D inputs (batch, channels, H, W)
        if len(inputs.shape) != 4:
            raise ValueError(
                "Inputs should be a 4D tensor (batch, channels, height, width)"
            )

        if len(targets.shape) != 1:
            raise ValueError("Targets should be a 1D tensor or array of class indices")

        if inputs.shape[0] != targets.shape[0]:
            raise ValueError("Batch size of inputs and targets must match")

        # Additional validation can be added here or delegated to backend

    def generate(self, model: Any, inputs: Any, targets: Any) -> Tuple[Any, Any]:
        """
        Generate adversarial examples using the implemented attack method.

        This is the main interface for running adversarial attacks. It handles
        input validation, model preparation, and orchestrates the attack process
        by calling the abstract `_perturb` method implemented by subclasses.

        The method follows this workflow:
        1. Validate input parameters
        2. Set model to evaluation mode
        3. Enable gradients on inputs
        4. Call the attack-specific perturbation logic
        5. Return adversarial examples and perturbation

        Args:
            model (Any): The model to attack. Must be a callable that takes
                        inputs and returns outputs.
            inputs (Any): Input batch tensor. Should be 4D with shape
                         (batch_size, channels, height, width).
            targets (Any): Target labels for targeted attacks or true labels
                          for untargeted attacks. Should be 1D tensor/array.

        Returns:
            Tuple[Any, Any]: A tuple containing:
                - adversarial_inputs: The adversarially perturbed input tensor
                - perturbation: The perturbation that was applied to create
                               the adversarial examples

        Raises:
            ValueError: If input validation fails.
            RuntimeError: If the attack fails to generate valid examples.

        Example:
            >>> attack = FGSMAttack(backend, epsilon=0.1)
            >>> adv_inputs, noise = attack.generate(model, inputs, targets)
            >>> print(f"Generated adversarial examples with {noise.shape} perturbation")
        """
        self._validate_inputs(model, inputs, targets)

        if self.verbose:
            print(
                f"Starting {'targeted' if self.targeted else 'untargeted'} attack with epsilon={self.epsilon}"
            )

        self.backend.model_eval(model)

        inputs_adv = self.backend.requires_grad(inputs)

        adv_images, perturbation = self._perturb(model, inputs_adv, targets)

        if self.verbose:
            print("Attack completed.")

        return adv_images, perturbation

    @abstractmethod
    def _perturb(self, model: Any, inputs: Any, targets: Any) -> Tuple[Any, Any]:
        """
        Implement attack-specific perturbation logic.

        This is an abstract method that must be implemented by all concrete
        attack classes. It contains the core algorithm for generating adversarial
        examples, such as computing gradients, applying perturbations, and
        ensuring constraints are satisfied.

        The method receives inputs with gradients enabled and should return
        both the adversarial examples and the perturbation that was applied.
        The perturbation is useful for analysis and visualization.

        Args:
            model (Any): The model to attack. This is typically in evaluation mode
                        and should be callable with the given inputs.
            inputs (Any): Input tensor with gradients enabled. The tensor should
                         have requires_grad=True to allow gradient computation.
            targets (Any): Target labels for targeted attacks or true labels
                          for untargeted attacks. Used in loss computation.

        Returns:
            Tuple[Any, Any]: A tuple containing:
                - adversarial_examples: The adversarially perturbed input tensor
                                       with the same shape as inputs
                - perturbation: The perturbation tensor that was applied to
                               create the adversarial examples

        Note:
            - This method is called by the `generate` method after input validation
              and model preparation
            - The inputs tensor already has gradients enabled
            - The model is already set to evaluation mode
            - Implementations should handle any framework-specific operations
              through the backend interface
            - The perturbation should be detached from the computation graph
              if it's not needed for further computation

        Example Implementation:
            >>> def _perturb(self, model, inputs, targets):
            >>>     # Compute model outputs
            >>>     outputs = model(inputs)
            >>>
            >>>     # Compute loss
            >>>     loss = self.loss_fn(outputs, targets)
            >>>
            >>>     # Compute gradients
            >>>     self.backend.backward(loss)
            >>>
            >>>     # Apply perturbation
            >>>     perturbation = self.backend.grad_sign(inputs)
            >>>     adversarial = inputs - self.epsilon * perturbation
            >>>
            >>>     return adversarial, perturbation
        """
        pass
