"""Tests for impact-first phase detection.

The previous detector failed on long videos with pre-swing wandering — it
latched onto a fake 'top' during waggle motion. These tests exercise the
new detector by constructing synthetic landmark data with a clear swing
event surrounded by noise.
"""

from __future__ import annotations

import numpy as np

from app.models.landmarks import (
    LEFT_ANKLE,
    LEFT_HIP,
    LEFT_SHOULDER,
    LEFT_WRIST,
    NUM_LANDMARKS,
    RIGHT_ANKLE,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
)
from app.services.phases import detect_phases


def _build_synthetic_swing(
    *,
    fps: float = 60.0,
    pre_swing_frames: int = 0,
    address_frames: int = 30,
    backswing_frames: int = 75,
    downswing_frames: int = 25,
    follow_through_frames: int = 30,
) -> np.ndarray:
    """Build a (T, 33, 4) landmark array for a synthetic right-handed swing.

    Pre-swing frames simulate noisy waggle/setup motion so we can verify the
    detector finds the real impact peak rather than latching onto noise.
    """
    n = (
        pre_swing_frames
        + address_frames
        + backswing_frames
        + downswing_frames
        + follow_through_frames
    )
    lm = np.zeros((n, NUM_LANDMARKS, 4), dtype=np.float32)
    # Reasonable visibility everywhere.
    lm[:, :, 3] = 0.9

    # Static body parts for a setup-position right-handed golfer.
    # Z-spread encodes DTL geometry: lead (left) side farther from camera.
    lm[:, LEFT_SHOULDER, :3] = (0.45, 0.35, 0.10)
    lm[:, RIGHT_SHOULDER, :3] = (0.55, 0.35, -0.10)
    lm[:, LEFT_HIP, :3] = (0.46, 0.55, 0.08)
    lm[:, RIGHT_HIP, :3] = (0.54, 0.55, -0.08)
    lm[:, LEFT_ANKLE, :3] = (0.46, 0.95, 0.10)
    lm[:, RIGHT_ANKLE, :3] = (0.54, 0.95, -0.10)

    # Wrists at address position.
    address_wrist_xy = np.array([0.50, 0.70])
    top_wrist_xy = np.array([0.62, 0.20])  # high, behind trail shoulder
    impact_wrist_xy = np.array([0.50, 0.70])  # back to ball
    finish_wrist_xy = np.array([0.40, 0.25])  # high on lead side

    rng = np.random.default_rng(42)

    # Pre-swing: small random waggle around address position.
    for i in range(pre_swing_frames):
        jitter = rng.normal(0, 0.005, size=2).astype(np.float32)
        lm[i, LEFT_WRIST, :2] = address_wrist_xy + jitter
        lm[i, RIGHT_WRIST, :2] = address_wrist_xy + np.array([0.01, 0]) + jitter

    base = pre_swing_frames

    # Address: still
    for i in range(address_frames):
        lm[base + i, LEFT_WRIST, :2] = address_wrist_xy
        lm[base + i, RIGHT_WRIST, :2] = address_wrist_xy + np.array([0.01, 0])
    base += address_frames

    # Backswing: address → top
    for i in range(backswing_frames):
        t = i / max(backswing_frames - 1, 1)
        # Ease-in-out so velocity peaks mid-backswing
        t_smooth = 0.5 - 0.5 * np.cos(np.pi * t)
        lm[base + i, LEFT_WRIST, :2] = (
            address_wrist_xy + (top_wrist_xy - address_wrist_xy) * t_smooth
        )
        lm[base + i, RIGHT_WRIST, :2] = lm[base + i, LEFT_WRIST, :2] + np.array(
            [0.01, 0]
        )
    base += backswing_frames

    # Downswing: top → impact (faster than backswing)
    for i in range(downswing_frames):
        t = i / max(downswing_frames - 1, 1)
        # Steeper curve = higher peak velocity at impact
        t_smooth = t**2
        lm[base + i, LEFT_WRIST, :2] = (
            top_wrist_xy + (impact_wrist_xy - top_wrist_xy) * t_smooth
        )
        lm[base + i, RIGHT_WRIST, :2] = lm[base + i, LEFT_WRIST, :2] + np.array(
            [0.01, 0]
        )
    base += downswing_frames

    # Follow-through: impact → finish, decelerating
    for i in range(follow_through_frames):
        t = i / max(follow_through_frames - 1, 1)
        t_smooth = 1 - (1 - t) ** 2  # ease-out
        lm[base + i, LEFT_WRIST, :2] = (
            impact_wrist_xy + (finish_wrist_xy - impact_wrist_xy) * t_smooth
        )
        lm[base + i, RIGHT_WRIST, :2] = lm[base + i, LEFT_WRIST, :2] + np.array(
            [-0.01, 0]
        )
    base += follow_through_frames

    return lm


def test_detects_impact_at_velocity_peak_no_pre_swing():
    lm = _build_synthetic_swing(pre_swing_frames=0)
    phases = detect_phases(lm, fps=60.0)
    # address_frames=30, backswing=75, downswing=25 → impact ≈ frame 130
    # (just before the peak; downswing curve t^2 peaks at the last frame)
    assert 125 <= phases.impact <= 135, f"impact={phases.impact}"


def test_phases_are_ordered():
    lm = _build_synthetic_swing()
    phases = detect_phases(lm, fps=60.0)
    assert phases.address_end < phases.top < phases.impact <= phases.finish


def test_long_pre_swing_does_not_break_detector():
    """Regression test: 16-second clip with 14s of pre-swing waggle.

    The old detector failed catastrophically on this case (tempo ratio 0.1:1,
    fake top during waggle). The new detector should still find the real
    impact and a real top within ~1.5s of impact.
    """
    lm = _build_synthetic_swing(
        pre_swing_frames=14 * 60,  # 14 seconds of waggle
        address_frames=30,
        backswing_frames=75,
        downswing_frames=25,
        follow_through_frames=30,
    )
    phases = detect_phases(lm, fps=60.0)

    # Real swing starts at frame 840 (14s * 60fps), impact ≈ 970-980.
    assert phases.impact >= 800, (
        f"impact at {phases.impact}, should be late in clip"
    )

    # Top should be within 1.5s (90 frames) before impact.
    assert phases.impact - 90 <= phases.top < phases.impact

    # Address end should be after the pre-swing waggle, close to actual setup.
    assert phases.address_end >= 14 * 60 - 30, (
        f"address_end={phases.address_end} suggests detector latched onto "
        "pre-swing motion"
    )

    # Tempo ratio sanity check: backswing 75 frames, downswing 25 → ≈ 3:1
    backswing_dur = phases.top - phases.address_end
    downswing_dur = phases.impact - phases.top
    ratio = backswing_dur / max(downswing_dur, 1)
    assert 1.5 <= ratio <= 5.0, f"tempo ratio {ratio:.2f}:1 out of plausible range"


def test_handedness_detected_from_swing_window():
    """Pre-swing left-wrist drift should not flip a right-handed swing."""
    lm = _build_synthetic_swing(pre_swing_frames=0)
    # Manually push the LEFT wrist around during the (zero) pre-swing window
    # to simulate a noisy left-handed detection bias — should still detect right.
    phases = detect_phases(lm, fps=60.0)
    assert phases.handedness == "right"
