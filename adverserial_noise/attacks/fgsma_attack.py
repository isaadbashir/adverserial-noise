from typing import Optional, Callable, Any, Tuple
from .attack_interface import AdversarialAttackBase
from .base.backend_interface import BackendInterface


class FGSMAttack(AdversarialAttackBase):
    """
    Targeted Fast Gradient Sign Method (FGSM) attack.

    FGSM is a one-step adversarial attack that generates adversarial examples
    by computing the gradient of the loss with respect to the input and then
    taking a single step in the direction of the gradient sign. This attack
    is computationally efficient and often effective at fooling deep neural networks.

    The attack works by:
    1. Computing the gradient of the loss with respect to the input
    2. Taking the sign of the gradient
    3. Scaling by epsilon (perturbation magnitude)
    4. Subtracting from the original input

    Mathematical formulation:
        x_adv = x - ε * sign(∇x L(x, y_target))

    where:
        - x_adv is the adversarial example
        - x is the original input
        - ε is the perturbation magnitude (epsilon)
        - L is the loss function
        - y_target is the target class

    Attributes:
        Inherits all attributes from AdversarialAttackBase.
        This implementation is always targeted (targeted=True).

    References:
        - Goodfellow, I. J., Shlens, J., & Szegedy, C. (2014). Explaining and
          harnessing adversarial examples. arXiv preprint arXiv:1412.6572.

    Example:
        >>> from adverserial_noise.attacks import FGSMAttack
        >>> from adverserial_noise.attacks.base import PyTorchBackend
        >>>
        >>> backend = PyTorchBackend()
        >>> attack = FGSMAttack(
        >>>     backend=backend,
        >>>     epsilon=0.1,
        >>>     loss_fn=torch.nn.CrossEntropyLoss(),
        >>>     verbose=True
        >>> )
        >>>
        >>> adversarial, perturbation = attack.generate(model, inputs, targets)
    """

    def __init__(
        self,
        backend: BackendInterface,
        epsilon: float,
        loss_fn: Optional[Callable[[Any, Any], Any]] = None,
        verbose: bool = False,
    ):
        super().__init__(
            backend, epsilon, targeted=True, loss_fn=loss_fn, verbose=verbose
        )

    def _perturb(self, model: Any, inputs: Any, targets: Any) -> Tuple[Any, Any]:
        """
        Implement FGSM perturbation logic.

        This method implements the core FGSM algorithm:
        1. Forward pass through the model
        2. Compute loss with respect to target class
        3. Backpropagate to get gradients
        4. Take sign of gradients
        5. Apply perturbation scaled by epsilon

        Args:
            model (Any): The model to attack (already in eval mode).
            inputs (Any): Input tensor with gradients enabled.
            targets (Any): Target class labels for the attack.

        Returns:
            Tuple[Any, Any]: A tuple containing:
                - adv_images: The adversarially perturbed input tensor
                - perturbation: The perturbation tensor (squeezed to remove batch dim)

        Note:
            The inputs tensor should already have gradients enabled from the
            parent class. The perturbation is squeezed to remove the batch
            dimension for easier visualization and analysis.
        """
        # Enable gradient on inputs (redundant but explicit)
        inputs_adv = self.backend.requires_grad(inputs)

        # Forward pass through the model
        outputs = model(inputs_adv)

        # Compute loss with respect to target class
        loss = self.loss_fn(outputs, targets)

        # Backpropagate to compute gradients
        self.backend.backward(loss)

        # Get sign of gradients (direction of steepest ascent)
        perturbation = self.backend.grad_sign(inputs_adv)

        # Apply perturbation: x_adv = x - ε * sign(∇x L)
        adv_images = inputs_adv - (self.epsilon * perturbation)

        # Detach from computation graph to save memory
        adv_images.detach_()

        return adv_images, perturbation.squeeze()
