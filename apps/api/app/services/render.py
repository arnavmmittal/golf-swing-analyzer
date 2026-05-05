"""Render an annotated MP4 with skeleton overlay, phase markers, and angle readouts.

Output is a same-resolution video with:
- BlazePose skeleton drawn on every frame (colored by visibility)
- Phase label in the top-left ("ADDRESS", "BACKSWING", "DOWNSWING", "FOLLOW-THROUGH")
- Frame counter and timestamp
- Big phase markers ("TOP", "IMPACT") at key frames for ~6 frames each
- Spine line and shoulder/hip lines so users can see the angles being measured
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.models.landmarks import (
    LEFT_HIP,
    LEFT_SHOULDER,
    NOSE,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    midpoint,
)
from app.services.phases import Phases


# MediaPipe Pose connection pairs (subset relevant to golf — torso + limbs).
SKELETON_EDGES: list[tuple[int, int]] = [
    (11, 12),  # shoulders
    (11, 13), (13, 15),  # left arm
    (12, 14), (14, 16),  # right arm
    (11, 23), (12, 24),  # torso sides
    (23, 24),  # hips
    (23, 25), (25, 27), (27, 29), (27, 31),  # left leg
    (24, 26), (26, 28), (28, 30), (28, 32),  # right leg
]


def render_annotated(
    video_path: str | Path,
    output_path: str | Path,
    landmarks: np.ndarray,
    phases: Phases,
    fps: float,
) -> Path:
    """Render an annotated MP4 alongside the user's input video."""
    video_path = str(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    try:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame_idx >= len(landmarks):
                break

            _draw_skeleton(frame, landmarks[frame_idx], width, height)
            _draw_axes(frame, landmarks[frame_idx], width, height)
            _draw_phase_label(frame, frame_idx, phases, fps, width, height)
            _draw_phase_marker(frame, frame_idx, phases, width, height)

            writer.write(frame)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    return output_path


def _draw_skeleton(frame: np.ndarray, points: np.ndarray, w: int, h: int) -> None:
    for a, b in SKELETON_EDGES:
        if points[a, 3] < 0.3 or points[b, 3] < 0.3:
            continue
        pa = (int(points[a, 0] * w), int(points[a, 1] * h))
        pb = (int(points[b, 0] * w), int(points[b, 1] * h))
        cv2.line(frame, pa, pb, (255, 220, 80), 2)
    # Joints
    for i in range(33):
        if points[i, 3] < 0.3:
            continue
        c = (int(points[i, 0] * w), int(points[i, 1] * h))
        cv2.circle(frame, c, 4, (40, 40, 220), -1)


def _draw_axes(frame: np.ndarray, points: np.ndarray, w: int, h: int) -> None:
    """Draw spine line, shoulder line, hip line — the three axes we measure."""
    if all(points[i, 3] >= 0.3 for i in (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)):
        ls = (int(points[LEFT_SHOULDER, 0] * w), int(points[LEFT_SHOULDER, 1] * h))
        rs = (int(points[RIGHT_SHOULDER, 0] * w), int(points[RIGHT_SHOULDER, 1] * h))
        lh = (int(points[LEFT_HIP, 0] * w), int(points[LEFT_HIP, 1] * h))
        rh = (int(points[RIGHT_HIP, 0] * w), int(points[RIGHT_HIP, 1] * h))
        cv2.line(frame, ls, rs, (0, 255, 0), 2)
        cv2.line(frame, lh, rh, (0, 200, 200), 2)

        sm = midpoint(np.array(ls), np.array(rs)).astype(int)
        hm = midpoint(np.array(lh), np.array(rh)).astype(int)
        cv2.line(frame, tuple(sm), tuple(hm), (200, 80, 200), 2)


def _draw_phase_label(
    frame: np.ndarray,
    frame_idx: int,
    phases: Phases,
    fps: float,
    w: int,
    h: int,
) -> None:
    if frame_idx <= phases.address_end:
        label = "ADDRESS"
    elif frame_idx <= phases.top:
        label = "BACKSWING"
    elif frame_idx <= phases.impact:
        label = "DOWNSWING"
    else:
        label = "FOLLOW-THROUGH"

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (340, 75), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(
        frame, label, (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA,
    )
    timestamp = frame_idx / fps if fps > 0 else 0.0
    cv2.putText(
        frame, f"f{frame_idx}  {timestamp:.2f}s", (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA,
    )


def _draw_phase_marker(
    frame: np.ndarray,
    frame_idx: int,
    phases: Phases,
    w: int,
    h: int,
) -> None:
    """Big TOP / IMPACT label that fades after a few frames."""
    marker = None
    if abs(frame_idx - phases.top) <= 3:
        marker = "TOP"
        color = (60, 220, 60)
    elif abs(frame_idx - phases.impact) <= 3:
        marker = "IMPACT"
        color = (60, 60, 220)
    if marker is None:
        return

    text_size, _ = cv2.getTextSize(marker, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 4)
    x = (w - text_size[0]) // 2
    y = h // 4
    cv2.putText(frame, marker, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 4, cv2.LINE_AA)
