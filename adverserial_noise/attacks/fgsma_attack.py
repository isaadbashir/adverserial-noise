from typing import Optional, Callable, Any, Tuple
from .attack_interface import AdversarialAttackBase
from .base.backend_interface import BackendInterface


class FGSMAttack(AdversarialAttackBase):
    """
    Targeted Fast Gradient Sign Method (FGSM) attack.

    Generates adversarial examples that cause the model to classify inputs as the specified target class.
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
        # Enable gradient on inputs
        inputs_adv = self.backend.requires_grad(inputs)

        outputs = model(inputs_adv)

        loss = self.loss_fn(outputs, targets)

        self.backend.backward(loss)

        perturbation = self.backend.grad_sign(inputs_adv)

        adv_images = inputs_adv - (self.epsilon * perturbation)
        adv_images.detach_()

        return adv_images, perturbation.squeeze()
