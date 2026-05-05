"""Tests for 3D metric extraction.

The metrics were previously computed in 2D image space which foreshortened
rotational angles by ~5x for down-the-line footage. These tests construct
synthetic 3D landmarks and verify the metrics produce sane values.
"""

from __future__ import annotations

import math

import numpy as np

from app.models.landmarks import (
    LEFT_ANKLE,
    LEFT_ELBOW,
    LEFT_HIP,
    LEFT_SHOULDER,
    LEFT_WRIST,
    NOSE,
    NUM_LANDMARKS,
    RIGHT_ANKLE,
    RIGHT_ELBOW,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
)
from app.services.metrics import compute_metrics
from app.services.phases import Phases


def _empty_lm(n_frames: int) -> np.ndarray:
    """Allocate a (T, 33, 4) landmark array with full visibility."""
    lm = np.zeros((n_frames, NUM_LANDMARKS, 4), dtype=np.float32)
    lm[..., 3] = 1.0
    return lm


def _set_static_body(lm: np.ndarray) -> None:
    """Plant a generic right-handed body posture in DTL coordinates.

    DTL means the camera is behind-ball looking down the target line. For a
    righty:
      - Lead (left) side faces the target → larger z (farther from camera)
      - Trail (right) side faces away from target → smaller z (closer)
      - Vertical = y-axis, image-y grows downward
    """
    # Shoulders
    lm[:, LEFT_SHOULDER, :3] = (0.0, 0.30, 0.10)   # lead, farther
    lm[:, RIGHT_SHOULDER, :3] = (0.0, 0.30, -0.10)  # trail, closer
    # Hips
    lm[:, LEFT_HIP, :3] = (0.0, 0.55, 0.08)
    lm[:, RIGHT_HIP, :3] = (0.0, 0.55, -0.08)
    # Ankles (z-spread: lead foot farther, trail foot closer)
    lm[:, LEFT_ANKLE, :3] = (0.0, 0.95, 0.10)
    lm[:, RIGHT_ANKLE, :3] = (0.0, 0.95, -0.10)
    # Nose stays at center vertically above shoulders
    lm[:, NOSE, :3] = (0.0, 0.15, 0.0)


def test_x_factor_3d_recovers_45_degrees_from_dtl_view():
    """A 90° shoulder turn with 45° hip turn produces 45° X-Factor.

    The 2D-only implementation read this as ~0° because the rotation axis
    is aligned with the camera's view direction in DTL footage.
    """
    # 30 frames of static rotated pose so the time smoother (window=7) is a no-op.
    lm = _empty_lm(30)
    _set_static_body(lm)

    top = 15
    # Shoulders rotated 90°, hips 45° — applied to ALL frames.
    lm[:, LEFT_SHOULDER, :3] = (-0.10, 0.30, 0.0)
    lm[:, RIGHT_SHOULDER, :3] = (0.10, 0.30, 0.0)
    sin45 = math.sin(math.radians(45))
    cos45 = math.cos(math.radians(45))
    lm[:, LEFT_HIP, :3] = (-0.08 * sin45, 0.55, 0.08 * cos45)
    lm[:, RIGHT_HIP, :3] = (0.08 * sin45, 0.55, -0.08 * cos45)

    phases = Phases(address_end=5, top=top, impact=25, finish=29, handedness="right")
    metrics = compute_metrics(lm, phases, fps=60.0)

    # Expect ~45° X-Factor (90 shoulder - 45 hip).
    assert 40.0 <= metrics.x_factor <= 50.0, f"X-Factor = {metrics.x_factor}"


def test_hip_open_at_impact_recovers_40_degrees():
    """40° of hip rotation past square at impact reads as 40°, not 0°.

    Uses 40 frames so address (frame 5) and impact (frame 30) are well
    apart — the time smoother (window=7) sees a continuous regime at each.
    """
    lm = _empty_lm(40)
    _set_static_body(lm)
    sin40 = math.sin(math.radians(40))
    cos40 = math.cos(math.radians(40))
    # Frames 15-39: hips rotated 40° (covers the impact frame).
    lm[15:, LEFT_HIP, :3] = (-0.08 * sin40, 0.55, 0.08 * cos40)
    lm[15:, RIGHT_HIP, :3] = (0.08 * sin40, 0.55, -0.08 * cos40)

    phases = Phases(address_end=5, top=12, impact=30, finish=39, handedness="right")
    metrics = compute_metrics(lm, phases, fps=60.0)
    assert 35.0 <= metrics.hip_open_impact <= 45.0, (
        f"hip_open = {metrics.hip_open_impact}"
    )


def test_lead_arm_at_top_straight_arm_reads_near_180():
    """A perfectly straight lead arm at the top reads ~180° joint angle."""
    lm = _empty_lm(20)
    _set_static_body(lm)
    top = 10
    # Place lead shoulder, elbow, wrist on a straight line — across all frames.
    lm[:, LEFT_SHOULDER, :3] = (0.0, 0.30, 0.10)
    lm[:, LEFT_ELBOW, :3] = (0.10, 0.30, 0.10)
    lm[:, LEFT_WRIST, :3] = (0.20, 0.30, 0.10)

    phases = Phases(address_end=4, top=top, impact=15, finish=19, handedness="right")
    metrics = compute_metrics(lm, phases, fps=60.0)
    assert metrics.lead_arm_top >= 175.0, f"lead_arm_top = {metrics.lead_arm_top}"


def test_lead_arm_at_top_right_angle_reads_90():
    """A right-angle bend at the elbow reads ~90°."""
    lm = _empty_lm(10)
    _set_static_body(lm)
    top = 5
    # Broadcast the test pose across all frames so the time-axis smoother
    # (window=7) doesn't dilute the values at the target frame.
    lm[:, LEFT_SHOULDER, :3] = (0.0, 0.30, 0.0)
    lm[:, LEFT_ELBOW, :3] = (0.10, 0.30, 0.0)
    lm[:, LEFT_WRIST, :3] = (0.10, 0.20, 0.0)

    phases = Phases(address_end=2, top=top, impact=8, finish=9, handedness="right")
    metrics = compute_metrics(lm, phases, fps=60.0)
    assert 85.0 <= metrics.lead_arm_top <= 95.0


def test_head_sway_uses_body_frame_not_image_x():
    """Head moving 5cm along the body lateral axis reads correctly.

    From a DTL view, 'lateral' (toward/away from target) is the camera's
    z-axis, not its x-axis. The 2D implementation read image-x only and
    missed the actual sway.
    """
    lm = _empty_lm(20)
    _set_static_body(lm)
    # Shift the nose by 0.05 along the shoulder-line direction (which is the
    # lateral body axis in our setup, predominantly +z).
    shoulder_axis = lm[0, RIGHT_SHOULDER, :3] - lm[0, LEFT_SHOULDER, :3]
    shoulder_axis_unit = shoulder_axis / np.linalg.norm(shoulder_axis)
    shoulder_w = float(np.linalg.norm(shoulder_axis))

    # Move the nose by 25% of shoulder width along that axis at frame 10.
    lm[10:, NOSE, :3] = lm[0, NOSE, :3] + shoulder_axis_unit * 0.25 * shoulder_w

    phases = Phases(address_end=0, top=8, impact=15, finish=18, handedness="right")
    metrics = compute_metrics(lm, phases, fps=60.0)
    # Should read 0.25 (within rounding tolerance).
    assert 0.20 <= metrics.head_sway <= 0.30, f"head_sway = {metrics.head_sway}"


def test_weight_transfer_3d_at_lead_foot_reads_one():
    """COM directly over the lead foot reads as 1.0."""
    lm = _empty_lm(10)
    _set_static_body(lm)
    impact = 8
    # Broadcast hips on top of the lead ankle across all frames (smoother-safe).
    lm[:, LEFT_HIP, :3] = lm[0, LEFT_ANKLE, :3] + np.array([0, -0.4, 0])
    lm[:, RIGHT_HIP, :3] = lm[0, LEFT_ANKLE, :3] + np.array([0, -0.4, 0])

    phases = Phases(address_end=2, top=5, impact=impact, finish=9, handedness="right")
    metrics = compute_metrics(lm, phases, fps=60.0)
    assert metrics.weight_transfer >= 0.95, f"weight_transfer = {metrics.weight_transfer}"


def test_weight_transfer_3d_at_trail_foot_reads_zero():
    lm = _empty_lm(10)
    _set_static_body(lm)
    impact = 8
    lm[:, LEFT_HIP, :3] = lm[0, RIGHT_ANKLE, :3] + np.array([0, -0.4, 0])
    lm[:, RIGHT_HIP, :3] = lm[0, RIGHT_ANKLE, :3] + np.array([0, -0.4, 0])

    phases = Phases(address_end=2, top=5, impact=impact, finish=9, handedness="right")
    metrics = compute_metrics(lm, phases, fps=60.0)
    assert metrics.weight_transfer <= 0.05, f"weight_transfer = {metrics.weight_transfer}"


def test_shaft_lean_positive_when_arm_leans_toward_target():
    """Lead arm slanted toward the target (lead direction) gives positive lean."""
    lm = _empty_lm(10)
    _set_static_body(lm)
    impact = 8

    # World-coord convention: y is UP, so the wrist below the shoulder has
    # a SMALLER y. Lead direction is +z (toward target).
    lm[:, LEFT_SHOULDER, :3] = (0.0, 0.30, 0.10)
    lm[:, LEFT_WRIST, :3] = (0.0, -0.05, 0.18)  # below + toward target = positive lean
    phases = Phases(address_end=0, top=5, impact=impact, finish=9, handedness="right")
    metrics = compute_metrics(lm, phases, fps=60.0)
    assert metrics.shaft_lean_impact > 0, f"shaft_lean = {metrics.shaft_lean_impact}"


def test_shaft_lean_negative_when_arm_lags_behind_target():
    lm = _empty_lm(10)
    _set_static_body(lm)
    impact = 8
    lm[:, LEFT_SHOULDER, :3] = (0.0, 0.30, 0.10)
    lm[:, LEFT_WRIST, :3] = (0.0, -0.05, 0.02)  # below + away from target
    phases = Phases(address_end=0, top=5, impact=impact, finish=9, handedness="right")
    metrics = compute_metrics(lm, phases, fps=60.0)
    assert metrics.shaft_lean_impact < 0, f"shaft_lean = {metrics.shaft_lean_impact}"
