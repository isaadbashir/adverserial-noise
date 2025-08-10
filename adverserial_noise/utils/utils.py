# utils.py

from typing import Optional, Tuple, Union, List, Any

import numpy as np
from PIL import Image
import cv2

from adverserial_noise.attacks.attack import BackendTypes
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
import torch.nn.functional as F

try:
    import tensorflow as tf
except ImportError:
    tf = None

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def read_image(image_path: str) -> Image.Image:
    """
    Read image from file path using PIL (Python Imaging Library).

    This function loads an image from the specified file path and returns
    a PIL Image object. It supports various image formats including JPEG,
    PNG, BMP, TIFF, and others supported by PIL.

    Args:
        image_path (str): Path to the image file. Can be relative or absolute.

    Returns:
        PIL.Image.Image: Loaded image object.

    Raises:
        FileNotFoundError: If the image file cannot be found or opened.
        OSError: If the image file is corrupted or in an unsupported format.

    Example:
        >>> image = read_image("path/to/image.jpg")
        >>> print(f"Image size: {image.size}")
        >>> print(f"Image mode: {image.mode}")

    Note:
        This function uses PIL instead of OpenCV for better cross-platform
        compatibility and format support. The returned image is in RGB format
        by default.
    """
    image = Image.open(image_path)
    if image is None:
        raise FileNotFoundError(f"Failed to load image {image_path}")
    return image


def image_to_tensor(
    image: Union[np.ndarray[Any, Any], Image.Image],
    backend: str,
    size: Tuple[int, int] = (224, 224),
) -> Union[torch.Tensor, tf.Tensor]:
    """
    Convert image to backend-specific tensor normalized for ImageNet.

    This function preprocesses an input image for use with pre-trained models
    by resizing to the specified dimensions and applying ImageNet normalization.
    It supports both PIL Images and numpy arrays as input.

    The preprocessing pipeline includes:
    1. Resizing to the target dimensions
    2. Converting to tensor format
    3. Applying ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    4. Adding batch dimension

    Args:
        image (Union[np.ndarray, PIL.Image.Image]): Input image. Can be:
            - PIL.Image.Image: Image object
            - numpy.ndarray: Image array (will be converted to RGB if in BGR format)
        backend (str): Target backend framework. Currently supports:
            - 'pytorch': Returns PyTorch tensor
            - 'tensorflow': Returns TensorFlow tensor (if available)
        size (Tuple[int, int], optional): Target dimensions (height, width).
                                         Defaults to (224, 224) for ImageNet models.

    Returns:
        Union[torch.Tensor, tf.Tensor]: Preprocessed tensor with shape (1, C, H, W)
                                       for PyTorch or (1, H, W, C) for TensorFlow.

    Raises:
        ValueError: If unsupported backend is specified.
        RuntimeError: If TensorFlow backend is requested but not available.

    Example:
        >>> image = read_image("cat.jpg")
        >>> tensor = image_to_tensor(image, backend="pytorch", size=(256, 256))
        >>> print(f"Tensor shape: {tensor.shape}")
        >>> print(f"Tensor dtype: {tensor.dtype}")

    Note:
        - The returned tensor includes a batch dimension (batch size = 1)
        - ImageNet normalization is applied by default
        - For numpy arrays, BGR to RGB conversion is performed automatically
        - The function assumes RGB input images
    """
    if isinstance(image, np.ndarray):
        # Convert BGR to RGB for consistency
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

    if backend == BackendTypes.PYTORCH:
        transform = transforms.Compose(
            [
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD),
            ]
        )
        tensor = transform(image).unsqueeze(0)  # add batch
        return tensor

    else:
        raise ValueError(f"Unsupported backend: {backend}")


def tensor_to_numpy(
    tensor: Union[torch.Tensor, tf.Tensor], backend: str
) -> np.ndarray[Any, Any]:
    """
    Convert a batch tensor (1, C, H, W) or (1, H, W, C) to a numpy image (H, W, C) in [0,1].

    Args:
        tensor: Input tensor with batch dimension.
        backend: Backend string ('pytorch' or 'tensorflow').

    Returns:
        numpy.ndarray: Image array (H, W, 3) normalized to [0,1].
    """
    if backend == BackendTypes.PYTORCH:
        img = tensor.squeeze(0).detach().cpu()
        img = img.numpy()
        img = np.transpose(img, (1, 2, 0))  # CHW to HWC
        return img

    else:
        raise ValueError(f"Unsupported backend: {backend}")


def denormalize_image(img: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    img = (img * np.array(STD)) + np.array(MEAN)
    img = np.clip(img, 0, 1)
    return img


def visualize_attack(
    original: Union[torch.Tensor, tf.Tensor],
    adversarial: Union[torch.Tensor, tf.Tensor],
    noise: Union[torch.Tensor, tf.Tensor],
    probs: np.ndarray[Any, Any],
    predicted_class: str,
    true_class: Optional[str] = None,
    class_names: Optional[List[str]] = None,
    backend: str = BackendTypes.PYTORCH,
) -> None:
    """
    Visualize the results of an adversarial attack with comprehensive plots.

    This function creates a 4-panel visualization showing:
    1. Original image with true class label
    2. Adversarial noise/perturbation
    3. Adversarial image with predicted class
    4. Top-5 predicted class probabilities

    The visualization is useful for:
    - Comparing original vs. adversarial examples
    - Analyzing the perturbation patterns
    - Understanding model confidence changes
    - Debugging attack effectiveness

    Args:
        original (Union[torch.Tensor, tf.Tensor]): Original input tensor with batch dimension.
        adversarial (Union[torch.Tensor, tf.Tensor]): Adversarial tensor with batch dimension.
        noise (Union[torch.Tensor, tf.Tensor]): Perturbation/noise tensor with batch dimension.
        probs (np.ndarray): 1D array of predicted class probabilities.
        predicted_class (str): Class label predicted for the adversarial example.
        true_class (Optional[str], optional): True class label of the original image.
                                            Defaults to None.
        class_names (Optional[List[str]], optional): List of all class names for
                                                   probability visualization. If None,
                                                   indices are used. Defaults to None.
        backend (str, optional): Backend framework used ('pytorch' or 'tensorflow').
                                Defaults to BackendTypes.PYTORCH.

    Returns:
        None: Displays the visualization plot.

    Example:
        >>> visualize_attack(
        >>>     original=original_tensor,
        >>>     adversarial=adversarial_tensor,
        >>>     noise=noise_tensor,
        >>>     probs=probabilities,
        >>>     predicted_class="gibbon",
        >>>     true_class="giant panda",
        >>>     class_names=imagenet_classes
        >>> )

    Note:
        - All input tensors should have batch dimension (will be squeezed automatically)
        - Images are automatically denormalized from ImageNet normalization
        - Noise is scaled to [0,1] range for better visualization
        - The function uses matplotlib for plotting and displays the result
        - Top-5 predictions are shown by default
    """

    orig_img = tensor_to_numpy(original, backend)
    adv_img = tensor_to_numpy(adversarial, backend)
    noise_img = tensor_to_numpy(noise, backend)

    orig_img = denormalize_image(orig_img)
    adv_img = denormalize_image(adv_img)

    noise_img = noise_img * 0.5 + 0.5  # scale noise to [0,1]

    fig, axs = plt.subplots(1, 4, figsize=(18, 5))

    axs[0].imshow(orig_img)
    axs[0].set_title(f"Original Image\nTrue: {true_class or 'Unknown'}")
    axs[0].axis("off")

    axs[1].imshow(noise_img)
    axs[1].set_title("Adversarial Noise")
    axs[1].axis("off")

    axs[2].imshow(adv_img)
    axs[2].set_title(f"Adversarial Image\nPredicted: {predicted_class}")
    axs[2].axis("off")

    if class_names is None:
        class_names = [str(i) for i in range(len(probs))]

    topk = 5
    topk_indices = np.argsort(probs)[-topk:][::-1]
    topk_probs = probs[topk_indices]
    topk_names = [class_names[i] for i in topk_indices]

    axs[3].barh(topk_names[::-1], topk_probs[::-1])
    axs[3].set_xlim(0, 1)
    axs[3].set_title("Top Predicted Probabilities")

    plt.tight_layout()
    plt.show()


def load_imagenet_classes() -> List[str]:
    import urllib.request

    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    class_idx = []
    with urllib.request.urlopen(url) as f:
        for line in f:
            class_idx.append(line.decode("utf-8").strip())
    return class_idx


def get_pred_probs(
    model: Any, tensor: Union[torch.Tensor, tf.Tensor], backend: str
) -> np.ndarray:
    if backend == BackendTypes.PYTORCH:
        with torch.no_grad():
            output = model(tensor)
            probs = F.softmax(output, dim=1)
        return probs.squeeze(0).cpu().numpy()

    else:
        raise ValueError(f"Unsupported backend: {backend}")
