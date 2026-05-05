"""Swing phase segmentation from landmark time series.

We split a swing into six phases by analyzing wrist motion:

    Address ─► Takeaway ─► Top ─► Downswing ─► Impact ─► Finish

The detector is **impact-first**:

    1.  Impact = peak combined wrist speed (the unmistakable event)
    2.  Handedness = which wrist's path is longer in a window around impact
        (using the whole clip biases on long videos with pre-swing wandering)
    3.  Top    = lead wrist's highest position in [impact - 1.5s, impact - 0.1s]
    4.  Address end = walking BACKWARDS from top, the latest frame where
        combined wrist speed dropped below 15% of peak
    5.  Finish = walking forward from impact, first frame below 20% of peak

Why impact-first: an earlier version started from the address end and
searched forward. On long videos (e.g. 18s clips with 16s of pre-swing
setup), pre-swing wrist motion crossed the takeaway threshold and the
detector latched onto a fake 'top' during the waggle, producing tempo
ratios like 0.1:1 and hip-rotation values past anatomical limits.
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


def detect_phases(landmarks: np.ndarray, fps: float) -> Phases:
    """Split swing into Address / Takeaway / Top / Impact / Finish.

    `landmarks` shape: (T, 33, 4). Returns frame indices.
    """
    n = landmarks.shape[0]
    if n < 10:
        # Degenerate clip — return something sensible to avoid downstream errors.
        return Phases(0, max(n // 4, 0), max(n // 2, 1), max(n - 1, 1), "right")

    # Smoothed wrist trajectories and combined speed signal.
    lw_xy = smooth_savgol(landmarks[:, LEFT_WRIST, :2], window=9, order=2)
    rw_xy = smooth_savgol(landmarks[:, RIGHT_WRIST, :2], window=9, order=2)
    lw_speed = velocity(lw_xy, fps=fps)
    rw_speed = velocity(rw_xy, fps=fps)
    combined_speed = smooth_savgol(
        (lw_speed + rw_speed)[:, None], window=9, order=2
    )[:, 0]

    # 1. Impact: peak combined wrist speed.
    impact = int(np.argmax(combined_speed))
    peak_speed = float(combined_speed[impact])

    # 2. Handedness: from a window around impact, not the whole clip.
    swing_back = int(2.0 * fps)
    swing_start = max(0, impact - swing_back)
    lw_path = _total_path_length(landmarks[swing_start : impact + 1, LEFT_WRIST, :2])
    rw_path = _total_path_length(landmarks[swing_start : impact + 1, RIGHT_WRIST, :2])
    handedness: Handedness = "right" if rw_path >= lw_path else "left"
    lead_wrist_idx = LEFT_WRIST if handedness == "right" else RIGHT_WRIST

    # 3. Top: lead wrist's highest position (smallest y) in the
    # backswing window — typically 0.6-1.5s before impact for a real swing.
    top_search_start = max(0, impact - int(1.5 * fps))
    top_search_end = max(top_search_start + 1, impact - int(0.1 * fps))
    if top_search_end > top_search_start:
        wrist_y = landmarks[top_search_start:top_search_end, lead_wrist_idx, 1].copy()
        if len(wrist_y) >= 5:
            wrist_y = smooth_savgol(wrist_y[:, None], window=5, order=2)[:, 0]
        top = top_search_start + int(np.argmin(wrist_y))
    else:
        top = max(impact - max(int(0.5 * fps), 1), 0)

    # 4. Address end. Two issues to handle:
    #   (a) Wrist velocity dips to ~0 AT the top of backswing (direction
    #       reversal), so a naive "first quiet frame walking backwards from
    #       top" stops on that dip.
    #   (b) The backswing's peak velocity is much smaller than the
    #       downswing's, so a threshold based on overall peak speed is too
    #       aggressive and the entire backswing reads as "quiet" in places.
    #
    # We anchor the threshold to the backswing peak (much smaller than the
    # downswing peak), then do a two-phase walk: first skip backwards past
    # the top-of-backswing velocity dip until we hit an active frame, then
    # continue backwards until we hit quiet again — that's the takeaway
    # start, which we report as address_end.
    backswing_window = combined_speed[max(0, top - int(1.5 * fps)) : max(top, 1)]
    backswing_peak = (
        float(np.max(backswing_window)) if len(backswing_window) > 0 else peak_speed
    )
    takeaway_threshold = backswing_peak * 0.20

    address_end = 0
    i = top
    # Phase 1: walk backwards through the top-of-backswing quiet zone
    # until we find an active frame.
    while i >= 0 and combined_speed[i] < takeaway_threshold:
        i -= 1
    # Phase 2: continue walking backwards until we find quiet again.
    while i >= 0:
        if combined_speed[i] < takeaway_threshold:
            address_end = i
            break
        i -= 1

    # 5. Finish: walking forward from impact, first frame below 20% of peak.
    finish_threshold = peak_speed * 0.20
    finish = n - 1
    for i in range(impact + 1, n):
        if combined_speed[i] < finish_threshold:
            finish = i
            break

    # Sanity: enforce ordering with at least 1-frame gaps where possible.
    if top <= address_end:
        top = min(address_end + 1, n - 1)
    if impact <= top:
        impact = min(top + 1, n - 1)
    if finish < impact:
        finish = min(impact + 1, n - 1)

    return Phases(
        address_end=int(address_end),
        top=int(top),
        impact=int(impact),
        finish=int(finish),
        handedness=handedness,
    )


def detect_handedness(landmarks: np.ndarray) -> Handedness:
    """Whole-clip handedness fallback for callers that don't have phases yet.

    Prefer the swing-window-restricted version inside `detect_phases` — using
    the whole clip is biased by any pre/post-swing motion.
    """
    lw_path = _total_path_length(landmarks[:, LEFT_WRIST, :2])
    rw_path = _total_path_length(landmarks[:, RIGHT_WRIST, :2])
    return "right" if rw_path >= lw_path else "left"


def _total_path_length(xy: np.ndarray) -> float:
    if xy.shape[0] < 2:
        return 0.0
    diffs = np.diff(xy, axis=0)
    return float(np.linalg.norm(diffs, axis=-1).sum())
