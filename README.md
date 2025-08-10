# Adversarial Noise

A Python library for generating adversarial examples using various attack methods. This library provides a framework-agnostic interface for implementing and running adversarial attacks against machine learning models.

## Features

- **Framework Agnostic**: Works with PyTorch, TensorFlow, and other frameworks through backend interfaces
- **Multiple Attack Methods**: Currently supports FGSM (Fast Gradient Sign Method) with extensible architecture
- **Easy to Use**: Simple API for running attacks and visualizing results
- **Production Ready**: Type hints, comprehensive error handling, and modular design

## Installation

### Prerequisites

- Python 3.10 or higher
- PyTorch (for PyTorch backend)
- TensorFlow (optional, for TensorFlow backend)

### Install from source

```bash
git clone https://github.com/isaadbashir/adverserial-noise.git
cd adverserial-noise
pip install -e .
```

### Install dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

```python
import torch
import torchvision.models as models
from adverserial_noise.attacks import FGSMAttack
from adverserial_noise.attacks.base import PyTorchBackend
from adverserial_noise.utils import utils

# Load a pre-trained model
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.eval()

# Load and preprocess an image
image = utils.read_image("path/to/image.jpg")
image_tensor = utils.image_to_tensor(image, backend="pytorch")

# Create attack instance
backend = PyTorchBackend()
attack = FGSMAttack(
    backend=backend,
    epsilon=0.1,
    loss_fn=torch.nn.CrossEntropyLoss(),
    verbose=True
)

# Generate adversarial example
adversarial_image, perturbation = attack.generate(
    model=model,
    inputs=image_tensor,
    targets=torch.tensor([target_class])
)
```

## Project Structure

```
adverserial_noise/
├── attacks/                 # Attack implementations
│   ├── base/               # Backend interfaces
│   │   ├── backend_interface.py    # Abstract backend interface
│   │   └── pytorch_interface.py    # PyTorch implementation
│   ├── attack_interface.py # Base attack class
│   └── fgsma_attack.py     # FGSM attack implementation
├── utils/                   # Utility functions
│   └── utils.py            # Image processing and visualization
└── examples/                # Usage examples
    └── adverserial_fgsm.ipynb
```

## Core Components

### Attack Interface

The `AdversarialAttackBase` class provides a common interface for all adversarial attacks:

- **Framework Agnostic**: Works with any backend that implements `BackendInterface`
- **Configurable**: Supports targeted/untargeted attacks with custom loss functions
- **Extensible**: Easy to implement new attack methods

### Backend Interface

The `BackendInterface` abstracts framework-specific operations:

- **Tensor Operations**: Conversion, gradient handling, device management
- **Model Operations**: Evaluation mode, parameter management
- **Framework Independence**: Easy to add support for new frameworks

### Utility Functions

Comprehensive utilities for:

- **Image Processing**: Loading, preprocessing, and converting images
- **Visualization**: Comparing original vs. adversarial examples
- **Model Integration**: Getting predictions and probabilities

## Supported Attacks

### FGSM (Fast Gradient Sign Method)

A targeted adversarial attack that:

- Computes gradients with respect to input
- Applies perturbation in the direction of gradient sign
- Achieves high success rates with minimal perturbation

## Usage Examples

### Basic FGSM Attack

```python
from adverserial_noise.attacks import FGSMAttack
from adverserial_noise.attacks.base import PyTorchBackend

# Initialize attack
backend = PyTorchBackend()
attack = FGSMAttack(
    backend=backend,
    epsilon=0.1,
    loss_fn=torch.nn.CrossEntropyLoss()
)

# Generate adversarial example
adversarial, noise = attack.generate(model, inputs, targets)
```

### Custom Loss Function

```python
def custom_loss(outputs, targets):
    # Implement your custom loss function
    return torch.nn.functional.cross_entropy(outputs, targets)

attack = FGSMAttack(
    backend=backend,
    epsilon=0.1,
    loss_fn=custom_loss
)
```

### Visualization

```python
from adverserial_noise.utils import utils

utils.visualize_attack(
    original=inputs,
    adversarial=adversarial,
    noise=noise,
    probs=probabilities,
    predicted_class="target_class",
    true_class="original_class"
)
```