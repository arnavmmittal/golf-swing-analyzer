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
    LEFT_ANKLE,
    LEFT_SHOULDER,
    LEFT_WRIST,
    RIGHT_ANKLE,
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


def detect_phases(
    landmarks: np.ndarray,
    fps: float,
    handedness_override: Handedness | None = None,
    world_landmarks: np.ndarray | None = None,
) -> Phases:
    """Split swing into Address / Takeaway / Top / Impact / Finish.

    `landmarks` shape: (T, 33, 4) — image-coord landmarks. Used for wrist
    velocity profile (only x,y matters here, image-relative units are fine).

    `world_landmarks` shape: (T, 33, 4) — metric world coordinates. Used
    for 3D handedness detection signals. Falls back to image landmarks
    if not provided.

    `handedness_override`: skip auto-detection and use this handedness.
    Recommended whenever the user knows their handedness — auto-detection
    is geometric and has failure modes on unusual camera angles.
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

    # 1a. Approximate impact: peak combined wrist speed in image coords.
    # The velocity peak typically lands 3-5 frames AFTER ball contact —
    # we'll refine this once handedness is known.
    impact_approx = int(np.argmax(combined_speed))
    peak_speed = float(combined_speed[impact_approx])
    impact = impact_approx

    # 2. Handedness: use override if supplied, else multi-signal vote.
    swing_back = int(2.0 * fps)
    swing_start = max(0, impact - swing_back)
    if handedness_override is not None:
        handedness: Handedness = handedness_override
    else:
        handedness = _detect_handedness_from_signals(
            world_landmarks if world_landmarks is not None else landmarks,
            swing_start=swing_start,
            impact=impact,
        )
    lead_wrist_idx = LEFT_WRIST if handedness == "right" else RIGHT_WRIST

    # 3. Top: lead wrist's highest position (smallest y) in the
    # backswing window — typically 0.6-1.5s before impact for a real swing.
    # Use the velocity-peak-based `impact_approx` here (NOT the refined
    # impact below). impact_approx is slightly late, which is fine: the
    # top-search window stays large enough to include the actual top.
    top_search_start = max(0, impact_approx - int(1.5 * fps))
    top_search_end = max(top_search_start + 1, impact_approx - int(0.1 * fps))
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

    # 4b. Refine impact: in world coords, the lead wrist reaches its lowest
    # point (most negative Y, closest to ground) at ball contact. The
    # velocity peak we anchored on is typically 3-5 frames AFTER that, in
    # the follow-through. We search a small window around the velocity peak.
    # Done AFTER top detection (which uses impact_approx) so that the
    # refinement only affects impact-frame body-position readings, not the
    # tempo measurement.
    if world_landmarks is not None:
        lookback = max(int(0.10 * fps), 3)  # ~6 frames at 60fps
        lookahead = max(int(0.05 * fps), 1)
        lo = max(top + 1, impact_approx - lookback)
        hi = min(n, impact_approx + lookahead + 1)
        if hi > lo:
            wrist_y_world = world_landmarks[lo:hi, lead_wrist_idx, 1]
            impact = lo + int(np.argmin(wrist_y_world))

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

    Prefer the multi-signal version inside `detect_phases` (or supplying an
    explicit override) — wrist path length is a poor signal because both
    wrists are on the club and travel similar distances.
    """
    lw_path = _total_path_length(landmarks[:, LEFT_WRIST, :2])
    rw_path = _total_path_length(landmarks[:, RIGHT_WRIST, :2])
    return "right" if rw_path >= lw_path else "left"


def _detect_handedness_from_signals(
    landmarks: np.ndarray, *, swing_start: int, impact: int
) -> Handedness:
    """Multi-signal handedness detection using 3D landmarks.

    Two signals voted:

    1.  Foot z-coordinate at swing-start (i.e. address). For DTL footage,
        the lead foot (which is on the target side) is FARTHER from the
        camera than the trail foot. MediaPipe's z is signed depth from
        the hip midpoint, smaller = closer to camera. So:
            righty in DTL: z(left_ankle) > z(right_ankle)

    2.  Wrist y-coordinate at address. The lead hand grips above the trail
        hand on the club, so the lead wrist sits slightly higher (smaller
        image y) than the trail wrist:
            righty: y(left_wrist) < y(right_wrist)

    Each signal votes "right" or "left"; we return the majority. If they
    disagree we fall back to the foot signal because it's stronger when the
    camera is in DTL position (the most useful angle for swing analysis).
    """
    # Use a small window centered just before the swing kicks off — this is
    # roughly the address position, before any takeaway motion.
    window_size = max(1, min(15, impact - swing_start))
    addr_window = slice(swing_start, swing_start + window_size)

    # Use median across the address window for robustness against single-frame
    # pose detection errors.
    z_left_ankle = float(np.median(landmarks[addr_window, LEFT_ANKLE, 2]))
    z_right_ankle = float(np.median(landmarks[addr_window, RIGHT_ANKLE, 2]))
    foot_vote: Handedness = "right" if z_left_ankle > z_right_ankle else "left"

    y_left_wrist = float(np.median(landmarks[addr_window, LEFT_WRIST, 1]))
    y_right_wrist = float(np.median(landmarks[addr_window, RIGHT_WRIST, 1]))
    wrist_vote: Handedness = "right" if y_left_wrist < y_right_wrist else "left"

    if foot_vote == wrist_vote:
        return foot_vote
    # Tie-break with the foot signal — more reliable from the standard DTL angle.
    return foot_vote


def _total_path_length(xy: np.ndarray) -> float:
    if xy.shape[0] < 2:
        return 0.0
    diffs = np.diff(xy, axis=0)
    return float(np.linalg.norm(diffs, axis=-1).sum())
