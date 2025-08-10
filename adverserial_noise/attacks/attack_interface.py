from abc import ABC, abstractmethod
from typing import Optional, Callable, Any, Tuple
from .base.backend_interface import BackendInterface


class AdversarialAttackBase(ABC):
    """
    Framework-agnostic base class for adversarial attacks.

    Attributes:
        backend: BackendInterface instance for tensor/model operations.
        epsilon: Maximum perturbation magnitude.
        targeted: Whether attack is targeted or untargeted.
        loss_fn: Loss function to optimize.
        verbose: Enable verbose output.
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
        # Basic type and shape checks can be backend-specific or generic
        # For simplicity, here's a minimal generic check:
        if inputs is None or targets is None or model is None:
            raise ValueError("Model, inputs, and targets must not be None")

        # Example: input tensor shape check for 4D inputs (batch, channels, H, W)
        if len(inputs.shape) != 4:
            raise ValueError(
                "Inputs should be a 4D tensor (batch, channels, height, width)"
            )

        if len(targets.shape) != 1:
            raise ValueError("Targets should be a 1D tensor or array of class indices")

        if inputs.shape[0] != targets.shape[0]:
            raise ValueError("Batch size of inputs and targets must match")

        # You can add more checks or delegate to backend here

    def generate(self, model: Any, inputs: Any, targets: Any) -> Tuple[Any, Any]:
        """
        Generate adversarial examples.

        Args:
            model: The model to attack.
            inputs: Input batch tensor.
            targets: Target labels (for targeted attacks) or true labels.

        Returns:
            Adversarially perturbed inputs.
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
        Implement attack-specific perturbation logic here.

        Args:
            model: Model to attack.
            inputs: Inputs with gradient enabled.
            targets: Target or true labels.

        Returns:
            Adversarial examples tensor.
        """
        pass
