from typing import Any, Dict, Optional, Tuple, Union
from scipy.ndimage import gaussian_filter
import torch
import numpy as np
import torch.nn.functional as F


def RealisticCloudMaskFunc(
    input: torch.tensor,
    seed: Optional[Union[int, Tuple[int, ...]]] = None,
    clouds_attributes: Optional[Dict[str, Any]] = None,
):

    clouds_attributes = clouds_attributes or {}
    scale = clouds_attributes.get(
        "scale", 2.0
    )  # Default to 1.0 if 'scale' is not provided
    threshold = clouds_attributes.get("threshold", 0.5)  # Default to 0.5
    blur = clouds_attributes.get("blur", 0.0)  # Default to 0.0
    cloud_size = clouds_attributes.get("cloud_size", 0.3)  # Default to 0.3
    cloud_count = clouds_attributes.get("cloud_count", 10)  # Default to 10

    if len(input.shape) != 3:
        raise ValueError("Shape should have 4 dimensions: [channels, height, width]")

    if seed is not None:
        np.random.seed(seed)

    channels, height, width = input.shape
    # Generate the base mask for a single spatial shape (height, width)
    mask = np.zeros((height, width), dtype=np.float32)

    for _ in range(cloud_count):
        center_x = np.random.randint(0, width)
        center_y = np.random.randint(0, height)

        cloud_width = int(width * cloud_size)
        cloud_height = int(height * cloud_size)

        noise = np.random.rand(cloud_height, cloud_width)
        smooth_noise = gaussian_filter(noise, sigma=scale)
        cloud = smooth_noise > threshold

        x_start = max(center_x - cloud_width // 2, 0)
        y_start = max(center_y - cloud_height // 2, 0)
        x_end = min(center_x + cloud_width // 2, width)
        y_end = min(center_y + cloud_height // 2, height)

        mask[y_start:y_end, x_start:x_end] = np.maximum(
            mask[y_start:y_end, x_start:x_end],
            cloud[: y_end - y_start, : x_end - x_start],
        )

    if blur > 0:
        mask = gaussian_filter(mask, sigma=blur) > 0.5

    mask = 1 - mask
    # Expand the mask to match the batch and channel dimensions
    mask_to_apply = np.tile(mask, (channels, 1, 1))

    # Convert mask to tensor
    mask_to_apply = torch.tensor(mask_to_apply, dtype=torch.float32)

    # Apply the mask to the input image (masking the input)
    masked_input = input * mask_to_apply

    return masked_input, torch.tensor(mask, dtype=torch.float32)


def transform_tensor_image(tensor_image, target_size=64 * 3, crop_size=60 * 3):
    _, h, w = tensor_image.shape

    tensor_image = F.interpolate(
        tensor_image.unsqueeze(0),
        size=(target_size, target_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    start_x = torch.randint(0, target_size - crop_size + 1, (1,)).item()
    start_y = torch.randint(0, target_size - crop_size + 1, (1,)).item()
    tensor_image = tensor_image[
        :, start_y : start_y + crop_size, start_x : start_x + crop_size
    ]

    if torch.rand(1).item() < 0.5:
        tensor_image = torch.flip(tensor_image, dims=[1])

    if torch.rand(1).item() < 0.5:
        tensor_image = torch.flip(tensor_image, dims=[2])

    return tensor_image
