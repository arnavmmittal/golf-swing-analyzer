"""Biomechanical metric extraction from landmark + phase data.

We compute eight metrics that together describe a swing's quality:

  1. tempo_ratio       backswing_duration / downswing_duration
  2. x_factor          shoulder-hip rotation separation at top  (deg, 3D)
  3. hip_open_impact   hip rotation past square at impact       (deg, 3D)
  4. spine_stability   stdev of spine tilt across the swing     (deg, 3D)
  5. lead_arm_top      lead-arm angle at top of backswing       (deg, 3D)
  6. head_sway         lateral head displacement / shoulder W   (ratio, body frame)
  7. weight_transfer   COM lateral travel toward lead foot      (ratio, 3D)
  8. shaft_lean_impact lead arm vs vertical at impact           (deg, 3D)

All rotational metrics use MediaPipe's `pose_world_landmarks` — metric 3D
coordinates in meters with the origin at the hip center and Y-axis aligned
with gravity (Y-up convention). The image-coord landmarks have a noisy
learned z that gives spurious readings on rotation metrics; world
landmarks are calibrated and ~5-10x more stable.

World-coord conventions:
  x: subject's right direction (meters)
  y: up (gravity direction = -Y, meters)
  z: out of screen toward viewer (meters)

The "horizontal plane" for rotational metrics is the xz plane.
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
from app.services.geometry import (
    angle_between,
    joint_angle,
    project_onto,
    signed_angle_diff_deg,
    signed_xz_angle,
    smooth_savgol,
)
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
    """Compute all eight metrics. `landmarks` shape: (T, 33, 4).

    Should be called with WORLD landmarks (metric 3D coordinates from
    MediaPipe's pose_world_landmarks), not image landmarks.

    We smooth the landmark time series with a Savitzky-Golay filter before
    extracting single-frame readings, because pose detection during the
    high-velocity downswing produces frame-to-frame jitter that would
    dominate single-frame metric readings otherwise.
    """
    landmarks = _smooth_landmarks(landmarks)

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
        shaft_lean_impact=_shaft_lean_at_impact(
            landmarks, phases, lead_shoulder, lead_wrist, lead_hip, trail_hip
        ),
    )


def _smooth_landmarks(lm: np.ndarray) -> np.ndarray:
    """Savitzky-Golay smoothing across the time axis for each landmark coord.

    Reduces frame-to-frame jitter in pose detection at high-velocity frames,
    which would otherwise corrupt single-frame readings used by impact-time
    metrics (shaft lean, hip rotation at impact, etc.).
    """
    if lm.shape[0] < 7:
        return lm
    out = lm.copy()
    # Smooth x, y, z (skip visibility column).
    flat = out[:, :, :3].reshape(out.shape[0], -1)
    smoothed = smooth_savgol(flat, window=7, order=2)
    out[:, :, :3] = smoothed.reshape(out.shape[0], out.shape[1], 3)
    return out


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
    """Shoulder-hip separation at top of backswing (degrees).

    Computed as the difference in xz-plane (horizontal) angle between the
    shoulder line and the hip line. Using 3D landmarks means rotations that
    tilt away from the camera (the dominant rotation axis in DTL footage)
    are measured correctly instead of foreshortened to ~zero.
    """
    shoulder_vec = lm[top_idx, rs, :3] - lm[top_idx, ls, :3]
    hip_vec = lm[top_idx, rh, :3] - lm[top_idx, lh, :3]
    shoulder_angle = float(signed_xz_angle(shoulder_vec))
    hip_angle = float(signed_xz_angle(hip_vec))
    diff = abs(signed_angle_diff_deg(shoulder_angle, hip_angle))
    return float(diff)


def _hip_open_at_impact(
    lm: np.ndarray,
    phases: Phases,
    lh: int,
    rh: int,
) -> float:
    """How far the hips have rotated past square (address) at impact (degrees).

    Computed in 3D xz-plane so rotations facing toward/away from the camera
    register correctly.
    """
    addr = max(phases.address_end - 1, 0)
    hip_addr = lm[addr, rh, :3] - lm[addr, lh, :3]
    hip_imp = lm[phases.impact, rh, :3] - lm[phases.impact, lh, :3]
    a_addr = float(signed_xz_angle(hip_addr))
    a_imp = float(signed_xz_angle(hip_imp))
    diff = signed_angle_diff_deg(a_imp, a_addr)
    return float(abs(diff))


def _spine_stability(
    lm: np.ndarray,
    phases: Phases,
    ls: int,
    rs: int,
    lh: int,
    rh: int,
) -> float:
    """Stdev (deg) of spine tilt across address through impact.

    Spine tilt is the angle between the spine vector (hip-mid → shoulder-mid)
    and gravity (negative y). Computed in 3D so tilt forward/back is captured
    in addition to side-to-side.
    """
    a, b = phases.address_end, max(phases.impact + 1, phases.address_end + 2)
    b = min(b, lm.shape[0])
    seg = lm[a:b]
    shoulder_mid = (seg[:, ls, :3] + seg[:, rs, :3]) / 2
    hip_mid = (seg[:, lh, :3] + seg[:, rh, :3]) / 2
    spine_vec = shoulder_mid - hip_mid  # shape (T, 3)

    # In world coords, up is +Y. Tilt = angle from the up direction.
    up = np.array([0.0, 1.0, 0.0])
    tilts = angle_between(spine_vec, up)  # shape (T,)
    if len(tilts) >= 5:
        tilts = smooth_savgol(tilts[:, None], window=5, order=2)[:, 0]
    return float(np.std(tilts))


def _lead_arm_at_top(
    lm: np.ndarray,
    top_idx: int,
    shoulder: int,
    elbow: int,
    wrist: int,
) -> float:
    """Lead-arm joint angle at the top of backswing (deg), 3D.

    A straight arm is 180°. Tour pros hold their lead arm at 80-95° relative
    to the shoulder-elbow-wrist hinge — this is the joint angle at the elbow
    and a low value means a bent arm (collapsed swing arc).
    """
    return float(
        joint_angle(
            lm[top_idx, shoulder, :3],
            lm[top_idx, elbow, :3],
            lm[top_idx, wrist, :3],
        )
    )


def _head_sway(
    lm: np.ndarray,
    phases: Phases,
    ls: int,
    rs: int,
) -> float:
    """Lateral head travel relative to shoulder width — measured in body frame.

    The "lateral" direction is the shoulder-line at address. We project the
    nose's displacement from its address position onto that axis and take
    the peak-to-peak range. This is robust to camera angle: from any angle,
    we measure body-relative motion.

    The denominator is the median 3D shoulder width across the swing window
    so a single bad pose-detection frame can't make the ratio explode.
    """
    a, b = phases.address_end, max(phases.impact + 1, phases.address_end + 2)
    b = min(b, lm.shape[0])

    # Body lateral axis at address = shoulder line.
    shoulder_axis_3d = lm[phases.address_end, rs, :3] - lm[phases.address_end, ls, :3]
    norm = np.linalg.norm(shoulder_axis_3d)
    if norm < 1e-6:
        return 0.0
    shoulder_axis_unit = shoulder_axis_3d / norm

    # Nose displacement from address position, projected onto body lateral axis.
    nose_addr = lm[phases.address_end, NOSE, :3]
    nose_path = lm[a:b, NOSE, :3]
    displacements = nose_path - nose_addr
    projected = displacements @ shoulder_axis_unit  # shape (T,)

    travel = float(projected.max() - projected.min())

    # Median shoulder width across the swing window for the denominator.
    shoulder_widths = np.linalg.norm(
        lm[a:b, rs, :3] - lm[a:b, ls, :3], axis=-1
    )
    shoulder_w = float(np.median(shoulder_widths))
    if shoulder_w < 1e-6:
        return 0.0
    return travel / shoulder_w


def _weight_transfer(
    lm: np.ndarray,
    phases: Phases,
    lead_ankle: int,
    trail_ankle: int,
    lead_hip: int,
    trail_hip: int,
) -> float:
    """Weight transfer ratio at impact (0 = on trail foot, 1 = on lead foot).

    Projects the hip-midpoint onto the line from trail ankle to lead ankle
    in 3D. From any camera angle, this measures the body's center of mass
    position relative to the stance line correctly.
    """
    impact = phases.impact
    lead_a = lm[impact, lead_ankle, :3]
    trail_a = lm[impact, trail_ankle, :3]
    hip_mid = (lm[impact, lead_hip, :3] + lm[impact, trail_hip, :3]) / 2

    stance_vec = lead_a - trail_a
    stance_dist = float(np.linalg.norm(stance_vec))
    if stance_dist < 1e-6:
        return 0.5
    stance_unit = stance_vec / stance_dist

    # Project hip_mid relative to trail ankle onto the stance axis.
    relative = hip_mid - trail_a
    projection = float(np.dot(relative, stance_unit))
    ratio = projection / stance_dist
    return float(np.clip(ratio, 0.0, 1.0))


def _shaft_lean_at_impact(
    lm: np.ndarray,
    phases: Phases,
    lead_shoulder: int,
    lead_wrist: int,
    lead_hip: int,
    trail_hip: int,
) -> float:
    """Approximate shaft lean using the lead arm as a proxy for the club.

    Positive = leaning toward target (good, ~5-9° on tour).

    "Toward target" is determined per-user from the hip line at address —
    the lead-side direction is from trail hip to lead hip in 3D. This works
    regardless of camera angle and handedness, fixing the buggy hand-coded
    sign convention in the previous version.
    """
    impact = phases.impact
    sh = lm[impact, lead_shoulder, :3]
    wr = lm[impact, lead_wrist, :3]
    arm_vec = wr - sh

    # In world coords, gravity-down is -Y.
    down = np.array([0.0, -1.0, 0.0])
    raw_angle = float(angle_between(arm_vec, down))

    # Lead direction at address: from trail hip to lead hip.
    addr = phases.address_end
    lead_dir = lm[addr, lead_hip, :3] - lm[addr, trail_hip, :3]
    lead_norm = np.linalg.norm(lead_dir)
    if lead_norm < 1e-6:
        return raw_angle
    lead_unit = lead_dir / lead_norm

    # Project arm vector onto lead direction. If arm leans toward target, the
    # projection is positive. If it lags behind (scoop), negative.
    forward_component = float(np.dot(arm_vec, lead_unit))
    sign = 1.0 if forward_component > 0 else -1.0
    return raw_angle * sign
