"""Geometric primitives used by the metric extractors.

All functions operate on either NumPy arrays of shape (T, ...) for time
series or single-frame arrays. Angles are returned in degrees.
"""

from __future__ import annotations

import numpy as np


def angle_between(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Angle between two vectors in degrees. Broadcasts over leading axes."""
    v1n = _normalize(v1)
    v2n = _normalize(v2)
    cos = np.clip(np.sum(v1n * v2n, axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Interior angle at vertex `b` for the triangle a-b-c."""
    return angle_between(a - b, c - b)


def signed_2d_angle(v: np.ndarray) -> np.ndarray:
    """Angle of a 2D vector measured from +x axis, in degrees, range (-180, 180]."""
    return np.degrees(np.arctan2(v[..., 1], v[..., 0]))


def signed_xz_angle(v: np.ndarray) -> np.ndarray:
    """Signed angle of a 3D vector projected onto the xz plane (perpendicular to vertical).

    Useful for measuring rotations around the gravity axis — body rotations
    like X-Factor and hip rotation at impact.

    `v` shape: (..., 3). Returns degrees in (-180, 180].
    """
    return np.degrees(np.arctan2(v[..., 2], v[..., 0]))


def signed_angle_diff_deg(a: float, b: float) -> float:
    """Signed angle difference a - b, wrapped to (-180, 180]."""
    diff = a - b
    while diff > 180:
        diff -= 360
    while diff <= -180:
        diff += 360
    return diff


def project_onto(v: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Scalar projection of `v` onto unit vector `axis`."""
    axis_norm = _normalize(axis)
    return np.sum(v * axis_norm, axis=-1)


def smooth_savgol(series: np.ndarray, window: int = 7, order: int = 2) -> np.ndarray:
    """Savitzky-Golay smoothing across the time axis.

    Falls back to a moving average if SciPy is missing for any reason.
    """
    try:
        from scipy.signal import savgol_filter

        if len(series) < window:
            return series
        if window % 2 == 0:
            window += 1
        return savgol_filter(series, window_length=window, polyorder=order, axis=0)
    except Exception:
        return _moving_average(series, window)


def velocity(series: np.ndarray, fps: float) -> np.ndarray:
    """Per-frame velocity magnitude. `series` shape (T, D)."""
    diffs = np.diff(series, axis=0, prepend=series[:1])
    speed = np.linalg.norm(diffs, axis=-1)
    return speed * fps


def _normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n < eps, eps, n)


def _moving_average(series: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(series) < window:
        return series
    kernel = np.ones(window) / window
    if series.ndim == 1:
        return np.convolve(series, kernel, mode="same")
    smoothed = np.empty_like(series)
    for i in range(series.shape[1]):
        smoothed[:, i] = np.convolve(series[:, i], kernel, mode="same")
    return smoothed
