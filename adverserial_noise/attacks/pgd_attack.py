from typing import Optional, Callable, Any, Tuple
from .attack_interface import AdversarialAttackBase
from .base.backend_interface import BackendInterface


class PGDAttack(AdversarialAttackBase):
    """
    Targeted Projected Gradient Descent (PGD) attack.

    PGD is an iterative adversarial attack that applies FGSM multiple times with
    small step sizes, projecting the adversarial example back into the valid
    epsilon-ball around the original input after each step to ensure the
    perturbation is bounded.

    The attack works by:
    1. Starting from the original input (optionally with random initialization)
    2. Iteratively computing gradients of the loss w.r.t input
    3. Taking a step in the direction of the gradient sign scaled by step size
    4. Projecting back into the epsilon-ball around the original input
    5. Repeating for a specified number of iterations

    Mathematical formulation:
        x_0_adv = x + δ (random initialization within epsilon ball, optional)
        x_{k+1}_adv = Π_{x, ε} (x_k_adv - α * sign(∇_x L(x_k_adv, y_target)))

    where:
        - Π_{x, ε} is the projection operator onto the epsilon ball centered at x
        - α is the step size
        - k is the iteration index

    Attributes:
        Inherits all attributes from AdversarialAttackBase.
        This implementation is always targeted (targeted=True).

    References:
        - Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2017).
          Towards deep learning models resistant to adversarial attacks.
          arXiv preprint arXiv:1706.06083.

    Example:
        >>> from adverserial_noise.attacks import PGDAttack
        >>> from adverserial_noise.attacks.base import PyTorchBackend
        >>>
        >>> backend = PyTorchBackend()
        >>> attack = PGDAttack(
        >>>     backend=backend,
        >>>     epsilon=0.1,
        >>>     step_size=0.01,
        >>>     num_iterations=40,
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
        step_size: float,
        num_iterations: int,
        loss_fn: Optional[Callable[[Any, Any], Any]] = None,
        verbose: bool = False,
        random_start: bool = True,
    ):
        super().__init__(
            backend, epsilon, targeted=True, loss_fn=loss_fn, verbose=verbose
        )
        self.step_size = step_size
        self.num_iterations = num_iterations
        self.random_start = random_start

    def _perturb(self, model: Any, inputs: Any, targets: Any) -> Tuple[Any, Any]:
        """
        Implement PGD perturbation logic.

        Args:
            model (Any): The model to attack (already in eval mode).
            inputs (Any): Input tensor (no gradient enabled here).
            targets (Any): Target class labels for the attack.

        Returns:
            Tuple[Any, Any]: A tuple containing:
                - adv_images: The adversarially perturbed input tensor
                - perturbation: The perturbation tensor (squeezed to remove batch dim)
        """
        # Initialize adversarial images
        if self.random_start:
            # Random perturbation within epsilon ball
            perturbation = self.backend.uniform(
                inputs.shape, low=-self.epsilon, high=self.epsilon
            )
            adv_images = inputs + perturbation
            # adv_images = self.backend.clamp(adv_images, 0.0, 1.0)  # Assuming inputs normalized between 0 and 1
        else:
            adv_images = inputs.clone()

        for i in range(self.num_iterations):
            adv_images = self.backend.requires_grad(adv_images)

            outputs = model(adv_images)
            loss = self.loss_fn(outputs, targets)

            self.backend.backward(loss)

            grad_sign = self.backend.grad_sign(adv_images)

            # Take a step in the direction of the gradient sign scaled by step_size
            adv_images = adv_images - self.step_size * grad_sign

            # Project back into epsilon-ball of the original input
            perturbation = adv_images - inputs
            perturbation = self.backend.clamp(perturbation, -self.epsilon, self.epsilon)
            adv_images = inputs + perturbation

            # Clamp to valid data range
            # adv_images = self.backend.clamp(adv_images, 0.0, 1.0)

            if self.verbose:
                print(f"PGD iteration {i + 1}/{self.num_iterations}")

        adv_images.detach_()
        perturbation = (adv_images - inputs).squeeze()

        return adv_images, perturbation
