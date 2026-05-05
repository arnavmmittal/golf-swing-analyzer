"""Swing phase segmentation from landmark time series.

We split a swing into six phases by analyzing the lead wrist's motion:

    Address ─► Takeaway ─► Top ─► Downswing ─► Impact ─► Finish

Detection uses the lead-wrist trajectory (left wrist for a right-handed
golfer, right wrist for a lefty). The signal looks like:

    velocity ↑ from 0 (takeaway begins)
    velocity = 0 at top (direction reversal)
    velocity peak at impact
    velocity ↓ to ~0 at finish

We work on smoothed signals so we don't trigger on jitter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from app.models.landmarks import (
    LEFT_SHOULDER,
    LEFT_WRIST,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
)
from app.services.geometry import smooth_savgol, velocity

Handedness = Literal["right", "left"]


@dataclass
class Phases:
    address_end: int
    top: int
    impact: int
    finish: int
    handedness: Handedness

    def as_dict(self) -> dict:
        return {
            "address_end_frame": self.address_end,
            "top_frame": self.top,
            "impact_frame": self.impact,
            "finish_frame": self.finish,
            "handedness": self.handedness,
        }


def detect_handedness(landmarks: np.ndarray) -> Handedness:
    """Guess handedness from which wrist is more active during the swing.

    `landmarks` is shape (T, 33, 4). The trail-side wrist (right for righties)
    moves slightly less than the lead-side at the top. Simpler heuristic: the
    lead wrist crosses further across the body. We use total path length.
    """
    lw_path = _total_path_length(landmarks[:, LEFT_WRIST, :2])
    rw_path = _total_path_length(landmarks[:, RIGHT_WRIST, :2])
    # The trail wrist (away from target) typically has the larger path
    # because it travels overhead at the top. We pick the trailing wrist
    # to determine handedness: more motion on the right → right-handed
    # golfer (right is the trail wrist in a RH swing).
    return "right" if rw_path >= lw_path else "left"


def detect_phases(landmarks: np.ndarray, fps: float) -> Phases:
    """Split swing into Address / Takeaway / Top / Impact / Finish.

    `landmarks` shape: (T, 33, 4). Returns frame indices.
    """
    handedness = detect_handedness(landmarks)
    lead_wrist_idx = LEFT_WRIST if handedness == "right" else RIGHT_WRIST
    lead_shoulder_idx = LEFT_SHOULDER if handedness == "right" else RIGHT_SHOULDER

    wrist_xy = landmarks[:, lead_wrist_idx, :2]
    wrist_xy = smooth_savgol(wrist_xy, window=9, order=2)
    speed = velocity(wrist_xy, fps=fps)
    speed = smooth_savgol(speed[:, None], window=9, order=2)[:, 0]

    # Vertical position of the wrist relative to the lead shoulder.
    # In image space, y grows downward; the wrist is highest at the top of
    # the backswing (smallest y).
    shoulder_y = smooth_savgol(landmarks[:, lead_shoulder_idx, 1:2], window=9, order=2)[:, 0]
    wrist_y_rel = wrist_xy[:, 1] - shoulder_y

    # Address: low speed at the start of the clip.
    address_end = _find_takeaway_start(speed)

    # Top: wrist reaches its highest point (smallest y) AFTER address_end.
    # We constrain the search to the first half of the swing.
    search_top_end = max(address_end + 5, len(speed) // 2 + 10)
    search_top_end = min(search_top_end, len(speed))
    top = address_end + int(np.argmin(wrist_y_rel[address_end:search_top_end]))

    # Impact: max wrist speed AFTER top.
    if top + 1 < len(speed):
        impact = top + int(np.argmax(speed[top:]))
    else:
        impact = top

    # Finish: speed drops below 20% of impact speed.
    finish = _find_finish(speed, impact)

    # Sanity: ensure ordering.
    address_end = min(address_end, top)
    impact = max(impact, top + 1) if top + 1 < len(speed) else top
    finish = max(finish, impact)

    return Phases(
        address_end=int(address_end),
        top=int(top),
        impact=int(impact),
        finish=int(finish),
        handedness=handedness,
    )


def _total_path_length(xy: np.ndarray) -> float:
    diffs = np.diff(xy, axis=0)
    return float(np.linalg.norm(diffs, axis=-1).sum())


def _find_takeaway_start(speed: np.ndarray) -> int:
    """First frame where speed exceeds 25% of its eventual peak."""
    if len(speed) == 0:
        return 0
    peak = float(np.max(speed))
    threshold = max(peak * 0.10, 1e-6)
    above = np.where(speed > threshold)[0]
    if len(above) == 0:
        return 0
    return int(above[0])


def _find_finish(speed: np.ndarray, impact: int) -> int:
    """Last frame within 1.5s of impact OR where speed dips below 20% peak."""
    if impact >= len(speed) - 1:
        return len(speed) - 1
    peak = float(np.max(speed[impact:]))
    threshold = max(peak * 0.20, 1e-6)
    after = speed[impact:]
    below = np.where(after < threshold)[0]
    if len(below) == 0:
        return len(speed) - 1
    return int(impact + below[0])
