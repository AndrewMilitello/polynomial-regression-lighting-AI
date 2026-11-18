from pathlib import Path
from PIL import Image
import numpy as np

from lighting_ai.config import *


def prepare_training_data(normal_path: Path, lighting_path: Path, n_samples: int = training_size, sample_size: int = sample_size):
    # images for training and inferrence go here
    normal_image = Image.open(normal_path).convert("RGB")
    lighting_image = Image.open(lighting_path).convert("RGB")

    # resize image to fixed resolution
    if max(normal_image.size) > MAX_DIM:
        scale = MAX_DIM / max(normal_image.size)
        new_size = (int(normal_image.width * scale), int(normal_image.height * scale))
        normal_image = normal_image.resize(new_size, Image.LANCZOS)

    lighting_image = lighting_image.resize(normal_image.size, Image.LANCZOS)

    # optional gaussian filter possibly to smooth out noise in the images, I found it didn't help much
    # normal = gaussian_filter(normal, sigma=1)
    # lighting = gaussian_filter(lighting, sigma=1)

    width, height = normal_image.size
    normal, lighting = np.array(normal_image), np.array(lighting_image)

    y1_coords = np.random.randint(0, normal.shape[0] - sample_size, size=n_samples) # Changed
    x1_coords = np.random.randint(0, normal.shape[1] - sample_size, size=n_samples)

    x2_coords = x1_coords + sample_size
    y2_coords = y1_coords + sample_size
    coords = list(zip(x1_coords, x2_coords, y1_coords, y2_coords))
    middle = sample_size // 2

    # mapping batch inpts (sample size squared) to single pixel outputs (mid)
    x_train = (np.array([normal[y1:y2, x1:x2].reshape(-1) for x1, x2, y1, y2 in coords]) / 255)
    y_train = (np.array([lighting[y1+middle, x1+middle] for x1, x2, y1, y2 in coords]) / 255)

    # normalization: scale by train amplitude
    x_train = x_train * train_amplitude - (train_amplitude/2)
    y_train = y_train * train_amplitude - (train_amplitude/2)

    return x_train, y_train

def extract_normal_patches(normal, sample_size: int = sample_size):
    # extract image patches and scale like training input
    normal = np.asarray(normal)

    if normal.ndim == 2:
        normal = normal[..., np.newaxis]
    elif normal.ndim != 3:
        raise ValueError("normal must be a 2D or 3D NumPy array")

    height, width, channels = normal.shape
    if sample_size < 1:
        raise ValueError("sample_size must be at least 1")

    normalized = (normal / 255.0).astype(np.float32)

    shift_before = sample_size // 2
    shift_after = sample_size - 1 - shift_before
    padded = np.pad(
        normalized,
        ((shift_before, shift_after), (shift_before, shift_after), (0, 0)),
        mode="edge",
    )

    kernels = np.lib.stride_tricks.sliding_window_view(
        padded,
        (sample_size, sample_size),
        axis=(0, 1),
    )
    kernels = kernels.transpose(0, 1, 3, 4, 2)

    scaled_patches = kernels.reshape(height * width, -1)
    scaled_patches = scaled_patches * train_amplitude - (train_amplitude / 2)

    middle = sample_size // 2
    center = (middle * sample_size + middle) * channels

    return scaled_patches, center