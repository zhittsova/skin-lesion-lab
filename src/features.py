"""HSV histogram stuff."""

from collections.abc import Sequence

import cv2
import numpy as np


def compute_hsv_histogram(
    hsv_image: np.ndarray,
    h_bins: int = 64,
    s_bins: int = 32,
    v_bins: int = 32,
    normalize: bool = True,
) -> np.ndarray:
    """Make one HSV feature vector."""
    if hsv_image.dtype != np.uint8:
        hsv_image = (
            (hsv_image * 255).astype(np.uint8)
            if hsv_image.max() <= 1
            else hsv_image.astype(np.uint8)
        )

    h_channel = hsv_image[:, :, 0]
    s_channel = hsv_image[:, :, 1]
    v_channel = hsv_image[:, :, 2]

    h_hist = cv2.calcHist([h_channel], [0], None, [h_bins], [0, 180])
    s_hist = cv2.calcHist([s_channel], [0], None, [s_bins], [0, 256])
    v_hist = cv2.calcHist([v_channel], [0], None, [v_bins], [0, 256])

    feature_vector = np.concatenate([h_hist.ravel(), s_hist.ravel(), v_hist.ravel()])

    if normalize:
        feature_vector = feature_vector / (feature_vector.sum() + 1e-10)

    return feature_vector


def compute_hsv_histograms_batch(
    hsv_images: Sequence[np.ndarray],
    h_bins: int = 64,
    s_bins: int = 32,
    v_bins: int = 32,
    normalize: bool = True,
) -> np.ndarray:
    """Same thing, but for a list of images."""
    all_features = []
    for img in hsv_images:
        feat = compute_hsv_histogram(img, h_bins, s_bins, v_bins, normalize)
        all_features.append(feat)

    return np.array(all_features)


def get_feature_dimension(h_bins: int = 64, s_bins: int = 32, v_bins: int = 32) -> int:
    return h_bins + s_bins + v_bins


def normalize_features(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return X / norms


def standardize_features(
    X: np.ndarray, mean: np.ndarray | None = None, std: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mean is None:
        mean: np.ndarray = X.mean(axis=0)
    if std is None:
        std: np.ndarray = X.std(axis=0)

    std[std == 0] = 1

    X_standardized = ((X - mean) / std).astype(np.float64)

    return X_standardized, mean, std


def apply_standardization(
    X: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    std_safe = std.copy()
    std_safe[std_safe == 0] = 1
    return ((X - mean) / std_safe).astype(np.float64)
