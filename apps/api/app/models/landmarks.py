"""MediaPipe BlazePose landmark indices and helpers.

BlazePose returns 33 landmarks per frame. We name the ones we care about for
golf so the metric code reads in body-part terms instead of magic numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# MediaPipe Pose landmark indices.
NOSE = 0
LEFT_EYE_INNER = 1
LEFT_EYE = 2
LEFT_EYE_OUTER = 3
RIGHT_EYE_INNER = 4
RIGHT_EYE = 5
RIGHT_EYE_OUTER = 6
LEFT_EAR = 7
RIGHT_EAR = 8
MOUTH_LEFT = 9
MOUTH_RIGHT = 10
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_PINKY = 17
RIGHT_PINKY = 18
LEFT_INDEX = 19
RIGHT_INDEX = 20
LEFT_THUMB = 21
RIGHT_THUMB = 22
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32

NUM_LANDMARKS = 33


@dataclass
class FrameLandmarks:
    """Landmarks for a single frame.

    `points`: image landmarks, shape (33, 4), columns (x, y, z, visibility).
        x, y are normalized to image dimensions (0-1); z is a learned depth
        estimate scaled to image x. Used for visual rendering.

    `world_points`: world landmarks, shape (33, 4), columns (x, y, z, visibility).
        Metric 3D coordinates in meters, origin at hip center, Y-axis up
        (against gravity). Used for biomechanical metric extraction.
    """

    points: np.ndarray
    world_points: np.ndarray
    frame_index: int
    timestamp_s: float

    def visible(self, idx: int, threshold: float = 0.5) -> bool:
        return bool(self.points[idx, 3] >= threshold)

    def xy(self, idx: int) -> np.ndarray:
        return self.points[idx, :2]

    def xyz(self, idx: int) -> np.ndarray:
        return self.points[idx, :3]


def midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) / 2.0
