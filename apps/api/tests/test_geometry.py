"""Tests for the geometric primitives that the metric extractors depend on.

We test geometry separately from metrics because metrics build on these — a
bug in `joint_angle` would silently corrupt every X-Factor and lead-arm
calculation.
"""

from __future__ import annotations

import math

import numpy as np

from app.services.geometry import angle_between, joint_angle, signed_2d_angle


def test_angle_between_orthogonal_is_90():
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.0, 1.0])
    assert math.isclose(float(angle_between(v1, v2)), 90.0, abs_tol=1e-6)


def test_angle_between_parallel_is_0():
    v1 = np.array([2.0, 0.0])
    v2 = np.array([5.0, 0.0])
    assert math.isclose(float(angle_between(v1, v2)), 0.0, abs_tol=1e-6)


def test_angle_between_antiparallel_is_180():
    v1 = np.array([1.0, 0.0])
    v2 = np.array([-3.0, 0.0])
    assert math.isclose(float(angle_between(v1, v2)), 180.0, abs_tol=1e-4)


def test_joint_angle_right_angle_at_b():
    # a-b-c where the angle at b is 90 degrees
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 0.0])
    c = np.array([0.0, 1.0])
    assert math.isclose(float(joint_angle(a, b, c)), 90.0, abs_tol=1e-6)


def test_signed_2d_angle_quadrants():
    assert math.isclose(float(signed_2d_angle(np.array([1.0, 0.0]))), 0.0, abs_tol=1e-6)
    assert math.isclose(float(signed_2d_angle(np.array([0.0, 1.0]))), 90.0, abs_tol=1e-6)
    assert math.isclose(float(signed_2d_angle(np.array([-1.0, 0.0]))), 180.0, abs_tol=1e-6)
    assert math.isclose(float(signed_2d_angle(np.array([0.0, -1.0]))), -90.0, abs_tol=1e-6)
