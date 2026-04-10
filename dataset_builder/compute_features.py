"""Statistical feature extraction for core SEM image prompt generation.

The extractor summarizes each original image with three complementary views:
global shape and scale, color/intensity distribution, and edge/gradient structure.
These descriptors are later serialized into the prompt sent to the language model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
from PIL import Image

_GRAY_HIST_BINS = 256
_GRAY_HIST_RANGE = (0, 255)
_CANNY_LOW_THRESHOLD = 80
_CANNY_HIGH_THRESHOLD = 160
_SOBEL_KERNEL_SIZE = 3


def load_rgb_image(image_path: Path) -> np.ndarray:
    """Load the original image in RGB order for statistical analysis."""
    return np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert RGB data to luminance space for intensity and edge statistics."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def compute_intensity_stats(gray_image: np.ndarray) -> Dict[str, float]:
    """Summarize the grayscale histogram with robust descriptive statistics."""
    normalized = gray_image.astype(np.float32) / 255.0
    histogram, _ = np.histogram(gray_image, bins=_GRAY_HIST_BINS, range=_GRAY_HIST_RANGE, density=True)
    histogram = histogram[histogram > 0]
    entropy = float(-(histogram * np.log2(histogram)).sum()) if histogram.size else 0.0

    return {
        "mean_intensity": round(float(normalized.mean()), 6),
        "std_intensity": round(float(normalized.std()), 6),
        "min_intensity": round(float(normalized.min()), 6),
        "max_intensity": round(float(normalized.max()), 6),
        "median_intensity": round(float(np.median(normalized)), 6),
        "p10_intensity": round(float(np.percentile(normalized, 10)), 6),
        "p90_intensity": round(float(np.percentile(normalized, 90)), 6),
        "entropy": round(entropy, 6),
    }


def compute_edge_density(gray_image: np.ndarray) -> float:
    """Estimate structural complexity via the fraction of Canny edge pixels."""
    edges = cv2.Canny(gray_image, _CANNY_LOW_THRESHOLD, _CANNY_HIGH_THRESHOLD)
    return float(np.count_nonzero(edges) / edges.size)


def compute_gradient_strength(gray_image: np.ndarray) -> float:
    """Measure average directional variation using first-order image gradients."""
    grad_x = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=_SOBEL_KERNEL_SIZE)
    grad_y = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=_SOBEL_KERNEL_SIZE)
    gradient_magnitude = cv2.magnitude(grad_x, grad_y)
    return float(np.mean(gradient_magnitude) / 255.0)


def compute_color_stats(rgb_image: np.ndarray) -> Dict[str, float]:
    """Capture per-channel brightness level and dispersion in normalized RGB space."""
    normalized = rgb_image.astype(np.float32) / 255.0
    channel_means = normalized.mean(axis=(0, 1))
    channel_stds = normalized.std(axis=(0, 1))

    return {
        "red_mean": round(float(channel_means[0]), 6),
        "green_mean": round(float(channel_means[1]), 6),
        "blue_mean": round(float(channel_means[2]), 6),
        "red_std": round(float(channel_stds[0]), 6),
        "green_std": round(float(channel_stds[1]), 6),
        "blue_std": round(float(channel_stds[2]), 6),
    }


def compute_features_from_image(image_path: Path) -> Dict[str, float]:
    """Compute the full statistical descriptor set from the original image only."""
    rgb_image = load_rgb_image(image_path)
    gray_image = to_grayscale(rgb_image)

    features: Dict[str, float] = {
        "width": int(rgb_image.shape[1]),
        "height": int(rgb_image.shape[0]),
        **compute_color_stats(rgb_image),
        **compute_intensity_stats(gray_image),
        "edge_density": round(compute_edge_density(gray_image), 6),
        "gradient_strength": round(compute_gradient_strength(gray_image), 6),
    }
    return features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute statistical image features for a rock core patch")
    parser.add_argument("--image", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = compute_features_from_image(args.image)
    print(json.dumps(features, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
