"""Pose extraction using MediaPipe BlazePose.

Reads a video file, runs pose detection on every frame, returns the time
series of landmarks. We use `model_complexity=2` (the heaviest BlazePose
model) because golf swings move fast and the lighter models drop frames at
the top and impact, where accuracy matters most.

We extract BOTH:
  - `points` (image landmarks): x, y normalized to image dims; z is a noisy
    learned depth estimate. Used for the visual renderer (skeleton overlay).
  - `world_points`: metric 3D coordinates in meters, with the origin at
    the hip center and Y-axis aligned with gravity (Y-up convention from
    MediaPipe). Used for biomechanical metric extraction.

The image-coord z is roughly 5-10x noisier than world-coord z. Using
`pose_world_landmarks` is essential for any metric that depends on real
3D geometry (X-Factor, hip rotation, shaft lean, etc.).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from app.models.landmarks import NUM_LANDMARKS, FrameLandmarks


def extract_pose(
    video_path: str | Path,
    *,
    model_complexity: int = 2,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> tuple[list[FrameLandmarks], dict]:
    """Run BlazePose over every frame.

    Returns (landmarks_per_frame, video_meta).
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        smooth_landmarks=True,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    frames: list[FrameLandmarks] = []
    try:
        frame_idx = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = pose.process(frame_rgb)

            points = np.zeros((NUM_LANDMARKS, 4), dtype=np.float32)
            world_points = np.zeros((NUM_LANDMARKS, 4), dtype=np.float32)
            if result.pose_landmarks is not None:
                for i, lm in enumerate(result.pose_landmarks.landmark):
                    points[i] = (lm.x, lm.y, lm.z, lm.visibility)
            if result.pose_world_landmarks is not None:
                for i, lm in enumerate(result.pose_world_landmarks.landmark):
                    world_points[i] = (lm.x, lm.y, lm.z, lm.visibility)

            frames.append(
                FrameLandmarks(
                    points=points,
                    world_points=world_points,
                    frame_index=frame_idx,
                    timestamp_s=frame_idx / fps,
                )
            )
            frame_idx += 1
    finally:
        cap.release()
        pose.close()

    meta = {
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "extracted_frames": len(frames),
    }
    return frames, meta


def stack_landmarks(frames: list[FrameLandmarks]) -> np.ndarray:
    """Stack per-frame image landmarks into shape (T, 33, 4)."""
    return np.stack([f.points for f in frames], axis=0)


def stack_world_landmarks(frames: list[FrameLandmarks]) -> np.ndarray:
    """Stack per-frame world (metric 3D) landmarks into shape (T, 33, 4)."""
    return np.stack([f.world_points for f in frames], axis=0)
