"""Tests for the multi-signal handedness detection.

The previous detector compared total wrist path length, which is a weak
signal because both wrists are on the same club and travel similar
distances. The new detector uses 3D foot position at address (lead foot
is farther from camera in DTL footage) plus wrist y-position (lead hand
grips above trail hand).
"""

from __future__ import annotations

import numpy as np

from app.models.landmarks import (
    LEFT_ANKLE,
    LEFT_SHOULDER,
    LEFT_WRIST,
    NUM_LANDMARKS,
    RIGHT_ANKLE,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
)
from app.services.phases import _detect_handedness_from_signals, detect_phases


def _build_address_lm(*, righty: bool, n_frames: int = 200) -> np.ndarray:
    """Build a (T, 33, 4) landmark array with a clear address-position signal.

    For a righty in DTL view:
      - Left ankle z > Right ankle z (lead foot is farther from camera)
      - Left wrist y < Right wrist y (lead hand higher on the grip)

    For a lefty: opposite of both.
    """
    lm = np.zeros((n_frames, NUM_LANDMARKS, 4), dtype=np.float32)
    lm[..., 3] = 1.0

    if righty:
        lm[:, LEFT_ANKLE, 2] = 0.10   # lead, farther
        lm[:, RIGHT_ANKLE, 2] = -0.10  # trail, closer
        lm[:, LEFT_WRIST, 1] = 0.55   # lead, higher (smaller y)
        lm[:, RIGHT_WRIST, 1] = 0.58  # trail, lower
    else:
        lm[:, RIGHT_ANKLE, 2] = 0.10
        lm[:, LEFT_ANKLE, 2] = -0.10
        lm[:, RIGHT_WRIST, 1] = 0.55
        lm[:, LEFT_WRIST, 1] = 0.58

    # Shoulders at standard width
    lm[:, LEFT_SHOULDER, :3] = (-0.1, 0.30, 0.10 if righty else -0.10)
    lm[:, RIGHT_SHOULDER, :3] = (0.1, 0.30, -0.10 if righty else 0.10)
    return lm


def test_detect_righty_from_3d_signals():
    lm = _build_address_lm(righty=True)
    h = _detect_handedness_from_signals(lm, swing_start=0, impact=100)
    assert h == "right"


def test_detect_lefty_from_3d_signals():
    lm = _build_address_lm(righty=False)
    h = _detect_handedness_from_signals(lm, swing_start=0, impact=100)
    assert h == "left"


def test_handedness_override_bypasses_auto_detection():
    """When the user supplies an override, auto-detection results are ignored."""
    # Build a clearly-righty configuration but override to "left".
    lm = _build_address_lm(righty=True, n_frames=300)
    # Inject minimal wrist motion so impact detection has a peak.
    lm[150:160, LEFT_WRIST, 0] = np.linspace(0, 0.5, 10)
    lm[150:160, RIGHT_WRIST, 0] = np.linspace(0, 0.5, 10)

    phases_auto = detect_phases(lm, fps=60.0)
    phases_forced = detect_phases(lm, fps=60.0, handedness_override="left")

    assert phases_auto.handedness == "right"  # auto correctly says righty
    assert phases_forced.handedness == "left"  # override wins
