"""Biomechanical metric extraction from landmark + phase data.

We compute eight metrics that together describe a swing's quality:

  1. tempo_ratio       backswing_duration / downswing_duration
  2. x_factor          shoulder-hip rotation separation at top  (deg)
  3. hip_open_impact   hip rotation past square at impact       (deg)
  4. spine_stability   stdev of spine tilt across the swing     (deg)
  5. lead_arm_top      lead-arm angle at top of backswing       (deg)
  6. head_sway         lateral head displacement / shoulder W   (ratio)
  7. weight_transfer   COM lateral travel toward lead foot      (ratio)
  8. shaft_lean_impact lead arm vs vertical at impact           (deg, +ve = forward)

All metrics are returned in a single dict keyed by metric name.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.models.landmarks import (
    LEFT_ANKLE,
    LEFT_ELBOW,
    LEFT_HIP,
    LEFT_SHOULDER,
    LEFT_WRIST,
    NOSE,
    RIGHT_ANKLE,
    RIGHT_ELBOW,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    midpoint,
)
from app.services.geometry import angle_between, joint_angle, signed_2d_angle, smooth_savgol
from app.services.phases import Phases


@dataclass
class Metrics:
    tempo_ratio: float
    x_factor: float
    hip_open_impact: float
    spine_stability: float
    lead_arm_top: float
    head_sway: float
    weight_transfer: float
    shaft_lean_impact: float

    def as_dict(self) -> dict:
        return {
            "tempo_ratio": round(self.tempo_ratio, 2),
            "x_factor": round(self.x_factor, 1),
            "hip_open_impact": round(self.hip_open_impact, 1),
            "spine_stability": round(self.spine_stability, 2),
            "lead_arm_top": round(self.lead_arm_top, 1),
            "head_sway": round(self.head_sway, 3),
            "weight_transfer": round(self.weight_transfer, 2),
            "shaft_lean_impact": round(self.shaft_lean_impact, 1),
        }


def compute_metrics(landmarks: np.ndarray, phases: Phases, fps: float) -> Metrics:
    """Compute all eight metrics. `landmarks` shape: (T, 33, 4)."""
    is_rh = phases.handedness == "right"
    lead_shoulder = LEFT_SHOULDER if is_rh else RIGHT_SHOULDER
    trail_shoulder = RIGHT_SHOULDER if is_rh else LEFT_SHOULDER
    lead_hip = LEFT_HIP if is_rh else RIGHT_HIP
    trail_hip = RIGHT_HIP if is_rh else LEFT_HIP
    lead_wrist = LEFT_WRIST if is_rh else RIGHT_WRIST
    lead_elbow = LEFT_ELBOW if is_rh else RIGHT_ELBOW
    lead_ankle = LEFT_ANKLE if is_rh else RIGHT_ANKLE
    trail_ankle = RIGHT_ANKLE if is_rh else LEFT_ANKLE

    return Metrics(
        tempo_ratio=_tempo_ratio(phases),
        x_factor=_x_factor(landmarks, phases.top, lead_shoulder, trail_shoulder, lead_hip, trail_hip),
        hip_open_impact=_hip_open_at_impact(landmarks, phases, lead_hip, trail_hip),
        spine_stability=_spine_stability(landmarks, phases, lead_shoulder, trail_shoulder, lead_hip, trail_hip),
        lead_arm_top=_lead_arm_at_top(landmarks, phases.top, lead_shoulder, lead_elbow, lead_wrist),
        head_sway=_head_sway(landmarks, phases, lead_shoulder, trail_shoulder),
        weight_transfer=_weight_transfer(landmarks, phases, lead_ankle, trail_ankle, lead_hip, trail_hip),
        shaft_lean_impact=_shaft_lean_at_impact(landmarks, phases.impact, lead_shoulder, lead_wrist, is_rh=is_rh),
    )


def _tempo_ratio(phases: Phases) -> float:
    backswing = max(phases.top - phases.address_end, 1)
    downswing = max(phases.impact - phases.top, 1)
    return float(backswing / downswing)


def _x_factor(
    lm: np.ndarray,
    top_idx: int,
    ls: int,
    rs: int,
    lh: int,
    rh: int,
) -> float:
    """Shoulder-hip separation at top of backswing (degrees, absolute value)."""
    shoulder_vec = lm[top_idx, rs, :2] - lm[top_idx, ls, :2]
    hip_vec = lm[top_idx, rh, :2] - lm[top_idx, lh, :2]
    shoulder_angle = signed_2d_angle(shoulder_vec)
    hip_angle = signed_2d_angle(hip_vec)
    diff = abs(shoulder_angle - hip_angle)
    if diff > 180:
        diff = 360 - diff
    return float(diff)


def _hip_open_at_impact(
    lm: np.ndarray,
    phases: Phases,
    lh: int,
    rh: int,
) -> float:
    """How far the hips have rotated past square (address) at impact."""
    addr = max(phases.address_end - 1, 0)
    hip_addr = lm[addr, rh, :2] - lm[addr, lh, :2]
    hip_imp = lm[phases.impact, rh, :2] - lm[phases.impact, lh, :2]
    a_addr = signed_2d_angle(hip_addr)
    a_imp = signed_2d_angle(hip_imp)
    diff = a_imp - a_addr
    while diff > 180:
        diff -= 360
    while diff < -180:
        diff += 360
    return float(abs(diff))


def _spine_stability(
    lm: np.ndarray,
    phases: Phases,
    ls: int,
    rs: int,
    lh: int,
    rh: int,
) -> float:
    """Stdev (deg) of spine tilt across address through impact."""
    a, b = phases.address_end, max(phases.impact + 1, phases.address_end + 2)
    b = min(b, lm.shape[0])
    seg = lm[a:b]
    shoulder_mid = midpoint(seg[:, ls, :2], seg[:, rs, :2])
    hip_mid = midpoint(seg[:, lh, :2], seg[:, rh, :2])
    spine_vec = shoulder_mid - hip_mid
    angles = signed_2d_angle(spine_vec)
    angles = smooth_savgol(angles[:, None], window=5, order=2)[:, 0]
    return float(np.std(angles))


def _lead_arm_at_top(
    lm: np.ndarray,
    top_idx: int,
    shoulder: int,
    elbow: int,
    wrist: int,
) -> float:
    """Lead-arm joint angle at the top of backswing (deg)."""
    return float(joint_angle(lm[top_idx, shoulder, :2], lm[top_idx, elbow, :2], lm[top_idx, wrist, :2]))


def _head_sway(
    lm: np.ndarray,
    phases: Phases,
    ls: int,
    rs: int,
) -> float:
    """Lateral head travel (max - min nose x) normalized by shoulder width.

    Below 0.05 is tour-level. Above 0.15 means the head is sliding off the ball.

    We use the MEDIAN shoulder width across the swing window as the
    denominator, not the value at a single frame. Single-frame normalization
    is brittle: if pose detection is poor at the address frame, the
    denominator can be near-zero and the ratio explodes.
    """
    a, b = phases.address_end, max(phases.impact + 1, phases.address_end + 2)
    b = min(b, lm.shape[0])
    nose_x = lm[a:b, NOSE, 0]
    shoulder_widths = np.linalg.norm(lm[a:b, rs, :2] - lm[a:b, ls, :2], axis=-1)
    shoulder_w = float(np.median(shoulder_widths))
    if shoulder_w < 1e-6:
        return 0.0
    travel = float(nose_x.max() - nose_x.min())
    return travel / shoulder_w


def _weight_transfer(
    lm: np.ndarray,
    phases: Phases,
    lead_ankle: int,
    trail_ankle: int,
    lead_hip: int,
    trail_hip: int,
) -> float:
    """Approximate weight transfer.

    Computes hip-midpoint x at impact relative to ankle midpoint. Closer to the
    lead foot = more weight transferred. We return a 0..1 ratio.
    """
    lead_x = float(lm[phases.impact, lead_ankle, 0])
    trail_x = float(lm[phases.impact, trail_ankle, 0])
    hip_mid_x = float(midpoint(lm[phases.impact, lead_hip, :2], lm[phases.impact, trail_hip, :2])[0])
    span = abs(lead_x - trail_x)
    if span < 1e-6:
        return 0.5
    # Project hip_mid_x onto the lead-trail axis. 0 = on trail foot, 1 = on lead.
    if lead_x < trail_x:
        ratio = (trail_x - hip_mid_x) / span
    else:
        ratio = (hip_mid_x - trail_x) / span
    return float(np.clip(ratio, 0.0, 1.0))


def _shaft_lean_at_impact(
    lm: np.ndarray,
    impact_idx: int,
    shoulder: int,
    wrist: int,
    *,
    is_rh: bool,
) -> float:
    """Approximate shaft lean using lead arm vs vertical at impact.

    Positive = leaning toward target (good, ~5-8° on tour).
    We use lead arm as a club proxy because we don't track the club head.
    """
    sh = lm[impact_idx, shoulder, :2]
    wr = lm[impact_idx, wrist, :2]
    arm_vec = wr - sh
    # Image y grows downward; vertical-down vector is (0, 1).
    vertical = np.array([0.0, 1.0])
    raw_angle = float(angle_between(arm_vec, vertical))
    # Sign: for RH golfer, target is +x; if wrist is left of (less x than)
    # shoulder, the arm leans toward target → positive lean.
    horizontal_component = wr[0] - sh[0]
    sign = -1.0 if (horizontal_component > 0) == is_rh else 1.0
    return raw_angle * sign
