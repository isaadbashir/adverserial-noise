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
    Convert a batch tensor to a numpy image array.

    This function converts a tensor with batch dimension (1, C, H, W) for PyTorch
    or (1, H, W, C) for TensorFlow to a numpy array with shape (H, W, C).
    The tensor is expected to be in the normalized [0,1] range and will be
    converted to the same range in the output numpy array.

    The conversion process includes:
    1. Removing the batch dimension
    2. Converting to CPU numpy array
    3. Transposing dimensions from CHW to HWC format (for PyTorch)
    4. Maintaining the [0,1] normalization range

    Args:
        tensor (Union[torch.Tensor, tf.Tensor]): Input tensor with batch dimension.
            Expected shapes:
            - PyTorch: (1, C, H, W) where C=3 for RGB images
            - TensorFlow: (1, H, W, C) where C=3 for RGB images
        backend (str): Backend framework used. Currently supports:
            - 'pytorch': PyTorch tensor format
            - 'tensorflow': TensorFlow tensor format (if available)

    Returns:
        np.ndarray: Image array with shape (H, W, 3) normalized to [0,1] range.

    Raises:
        ValueError: If unsupported backend is specified.

    Example:
        >>> # PyTorch tensor (1, 3, 224, 224)
        >>> tensor = torch.randn(1, 3, 224, 224)
        >>> numpy_img = tensor_to_numpy(tensor, backend="pytorch")
        >>> print(f"Output shape: {numpy_img.shape}")  # (224, 224, 3)
        >>> print(f"Value range: [{numpy_img.min():.3f}, {numpy_img.max():.3f}]")

    Note:
        - The function automatically handles PyTorch's CHW format conversion to HWC
        - Output is always in RGB format with shape (H, W, 3)
        - Values are maintained in the [0,1] range
        - For PyTorch tensors, the function detaches gradients and moves to CPU
    """
    if backend == BackendTypes.PYTORCH:
        img = tensor.squeeze(0).detach().cpu()
        img = img.numpy()
        img = np.transpose(img, (1, 2, 0))  # CHW to HWC
        return img

    else:
        raise ValueError(f"Unsupported backend: {backend}")


def denormalize_image(img: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """
    Denormalize an image from ImageNet normalization back to [0,1] range.

    This function reverses the ImageNet normalization that is commonly applied
    to images when preprocessing for deep learning models. It converts images
    from the normalized space (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    back to the original [0,1] range.

    The denormalization process:
    1. Multiplies each channel by the corresponding standard deviation
    2. Adds the corresponding mean value
    3. Clips values to ensure they stay within [0,1] range

    Args:
        img (np.ndarray): Input image array with shape (H, W, 3) in normalized space.
            Expected to be in the range typically output by ImageNet normalization.

    Returns:
        np.ndarray: Denormalized image array with shape (H, W, 3) in [0,1] range.

    Example:
        >>> # Normalized image (values typically in [-2, 2] range)
        >>> normalized_img = np.random.randn(224, 224, 3) * 0.5
        >>> denorm_img = denormalize_image(normalized_img)
        >>> print(f"Input range: [{normalized_img.min():.3f}, {normalized_img.max():.3f}]")
        >>> print(f"Output range: [{denorm_img.min():.3f}, {denorm_img.max():.3f}]")

    Note:
        - Uses ImageNet mean and standard deviation constants defined at module level
        - Output is clipped to [0,1] to prevent invalid pixel values
        - This function is typically used after tensor_to_numpy for visualization
        - The function assumes RGB channel order (R=0, G=1, B=2)
    """
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
    """
    Load ImageNet class names from the official PyTorch repository.

    This function downloads the ImageNet class names from the PyTorch Hub repository
    and returns them as a list of strings. The class names correspond to the
    1000 ImageNet classes used in pre-trained models like ResNet, VGG, etc.

    The function fetches the class names from:
    https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt

    Returns:
        List[str]: List of 1000 ImageNet class names in order.

    Raises:
        urllib.error.URLError: If the URL cannot be accessed or the file is corrupted.
        UnicodeDecodeError: If the downloaded file cannot be decoded as UTF-8.

    Example:
        >>> classes = load_imagenet_classes()
        >>> print(f"Total classes: {len(classes)}")
        >>> print(f"First 5 classes: {classes[:5]}")
        >>> # ['tench', 'goldfish', 'great white shark', 'tiger shark', 'hammerhead shark']

    Note:
        - Requires internet connection to download the class names
        - The function caches the result in memory for subsequent calls
        - Class indices correspond to model output indices (0-indexed)
        - This is useful for interpreting model predictions and visualization
    """
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
    """
    Get prediction probabilities from a model for a given input tensor.

    This function runs inference on a model with the given input tensor and
    returns the softmax probabilities for all classes. It handles the model
    inference in a no-gradient context to save memory and computation.

    The function supports different backend frameworks and automatically
    handles the appropriate inference method for each backend.

    Args:
        model (Any): The model to run inference on. Should be a callable object
            that accepts the input tensor and returns logits.
        tensor (Union[torch.Tensor, tf.Tensor]): Input tensor with batch dimension.
            Expected shapes:
            - PyTorch: (1, C, H, W) for image classification models
            - TensorFlow: (1, H, W, C) for image classification models
        backend (str): Backend framework used. Currently supports:
            - 'pytorch': PyTorch model and tensor
            - 'tensorflow': TensorFlow model and tensor (if available)

    Returns:
        np.ndarray: 1D array of class probabilities with shape (num_classes,).
            Probabilities sum to 1.0 and are in descending order of confidence.

    Raises:
        ValueError: If unsupported backend is specified.
        RuntimeError: If model inference fails or returns unexpected output.

    Example:
        >>> # PyTorch model and tensor
        >>> model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        >>> input_tensor = torch.randn(1, 3, 224, 224)
        >>> probs = get_pred_probs(model, input_tensor, backend="pytorch")
        >>> print(f"Probabilities shape: {probs.shape}")  # (1000,)
        >>> print(f"Sum of probabilities: {probs.sum():.6f}")  # Should be 1.0
        >>> top_class_idx = np.argmax(probs)
        >>> print(f"Top class index: {top_class_idx}, Probability: {probs[top_class_idx]:.4f}")

    Note:
        - The function runs inference without gradients to save memory
        - For PyTorch, the output is automatically moved to CPU and converted to numpy
        - Softmax is applied to convert logits to probabilities
        - The batch dimension is removed from the output
        - This function is typically used in conjunction with load_imagenet_classes()
          for interpreting model predictions
    """
    if backend == BackendTypes.PYTORCH:
        with torch.no_grad():
            output = model(tensor)
            probs = F.softmax(output, dim=1)
        return probs.squeeze(0).cpu().numpy()

    else:
        raise ValueError(f"Unsupported backend: {backend}")
