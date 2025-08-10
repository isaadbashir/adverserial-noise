from typing import Optional, Union, Any, Callable
import torch
import tensorflow as tf
from dataclasses import dataclass

from .base.pytorch_interface import PyTorchBackend
from .base.tensorflow_interface import TensorFlowBackend
from .fgsma_attack import FGSMAttack


@dataclass(frozen=True)
class BackendTypes:
    PYTORCH: str = "pytorch"
    TENSORFLOW: str = "tensorflow"

    @classmethod
    def values(cls) -> set[str]:
        return {cls.PYTORCH, cls.TENSORFLOW}


@dataclass(frozen=True)
class AttackTypes:
    FGSM: str = "fgsm"

    @classmethod
    def values(cls) -> set[str]:
        return {cls.FGSM}


SUPPORTED_BACKENDS = BackendTypes.values()
SUPPORTED_ATTACKS = AttackTypes.values()


class AdversarialAttack:
    """
    User-friendly interface to run adversarial attacks with minimal configuration.
    """

    def __init__(self, backend: Optional[str] = None) -> None:
        """
        Initialize the attack interface.

        Args:
            backend (str, optional): "pytorch" or "tensorflow". Auto-detected if None.
        """
        if backend is not None and backend.lower() not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported backend '{backend}'. Supported: {SUPPORTED_BACKENDS}"
            )
        self.backend_name = backend.lower() if backend else None

    def _detect_backend(self, model: Any) -> str:
        if self.backend_name:
            return self.backend_name
        if isinstance(model, torch.nn.Module):
            return BackendTypes.PYTORCH
        elif isinstance(model, (tf.Module, tf.keras.Model)):
            return BackendTypes.TENSORFLOW
        else:
            raise ValueError(
                "Cannot detect backend from model type. Please specify 'backend' explicitly."
            )

    def _create_backend(
        self, backend_name: str
    ) -> Union[PyTorchBackend, TensorFlowBackend]:
        if backend_name == BackendTypes.PYTORCH:
            return PyTorchBackend()
        elif backend_name == BackendTypes.TENSORFLOW:
            return TensorFlowBackend()
        else:
            raise ValueError(f"Unknown backend '{backend_name}'")

    def run_attack(
        self,
        model: Any,
        inputs: Any,
        targets: Any,
        attack_type: str = AttackTypes.FGSM,
        epsilon: float = 0.01,
        loss_fn: Optional[Callable[..., Any]] = None,
        verbose: bool = False,
    ) -> Any:
        """
        Run the specified adversarial attack on inputs.

        Args:
            model: Pretrained model (PyTorch or TensorFlow).
            inputs: Input tensor.
            targets: Target tensor.
            attack_type (str): One of SUPPORTED_ATTACKS.
            epsilon (float): Perturbation limit.
            alpha (float, optional): Step size (required for PGD).
            num_iterations (int, optional): Number of PGD iterations.
            random_start (bool): Whether PGD uses random start.
            loss_fn (callable, optional): Loss function.
            verbose (bool): Enable logging.

        Returns:
            Tensor of adversarial images.
        """
        backend_name = self._detect_backend(model)
        backend = self._create_backend(backend_name)

        attack_type = attack_type.lower()
        if attack_type not in SUPPORTED_ATTACKS:
            raise ValueError(
                f"Unsupported attack '{attack_type}'. Choose from {SUPPORTED_ATTACKS}"
            )

        if attack_type == AttackTypes.FGSM:
            attack = FGSMAttack(backend, epsilon, loss_fn=loss_fn, verbose=verbose)
        else:
            raise ValueError(f"Unsupported attack type: {attack_type}")

        return attack.generate(model, inputs, targets)
