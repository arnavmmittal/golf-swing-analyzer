"""Tests for the metric → score pipeline.

The scoring function is what turns raw biomechanics into the user-visible
0-100 grade, so we want strong coverage on the boundary conditions.
"""

from __future__ import annotations

from app.services.metrics import Metrics
from app.services.scoring import overall_score, score_metrics, top_faults


def _perfect_metrics() -> Metrics:
    return Metrics(
        tempo_ratio=3.0,
        x_factor=47.0,
        hip_open_impact=40.0,
        spine_stability=2.0,
        lead_arm_top=88.0,
        head_sway=0.03,
        weight_transfer=0.82,
        shaft_lean_impact=7.0,
    )


def _amateur_metrics() -> Metrics:
    return Metrics(
        tempo_ratio=4.5,        # rushed downswing
        x_factor=22.0,          # under-rotated torso
        hip_open_impact=15.0,   # hips not clearing
        spine_stability=6.0,    # posture changes
        lead_arm_top=60.0,      # collapsed lead arm
        head_sway=0.18,         # sliding off the ball
        weight_transfer=0.45,   # hanging back
        shaft_lean_impact=-2.0, # scooping
    )


def test_perfect_swing_scores_100_overall():
    scores = score_metrics(_perfect_metrics())
    assert overall_score(scores) == 100
    assert all(s.score == 100 for s in scores)
    assert all(s.fault is None for s in scores)


def test_amateur_swing_scores_low_overall():
    scores = score_metrics(_amateur_metrics())
    overall = overall_score(scores)
    assert 0 <= overall <= 50, f"expected low score, got {overall}"


def test_top_faults_are_lowest_scores():
    scores = score_metrics(_amateur_metrics())
    faults = top_faults(scores, n=3)
    assert len(faults) == 3
    # Faults must be sorted ascending by score (worst first)
    assert faults[0].score <= faults[1].score <= faults[2].score
    # Each top fault must actually be a fault, not a clean metric
    for f in faults:
        assert f.fault in ("low", "high")


def test_score_at_ideal_boundary_is_100():
    # tempo_ratio ideal_min is 2.8, so exactly 2.8 should be 100
    m = Metrics(
        tempo_ratio=2.8, x_factor=47.0, hip_open_impact=40.0,
        spine_stability=2.0, lead_arm_top=88.0, head_sway=0.03,
        weight_transfer=0.82, shaft_lean_impact=7.0,
    )
    scores = {s.name: s for s in score_metrics(m)}
    assert scores["tempo_ratio"].score == 100


def test_score_below_acceptable_is_zero():
    # x_factor acceptable_min is 30; anything <= 30 should be 0
    m = Metrics(
        tempo_ratio=3.0, x_factor=15.0, hip_open_impact=40.0,
        spine_stability=2.0, lead_arm_top=88.0, head_sway=0.03,
        weight_transfer=0.82, shaft_lean_impact=7.0,
    )
    scores = {s.name: s for s in score_metrics(m)}
    assert scores["x_factor"].score == 0
    assert scores["x_factor"].fault == "low"
