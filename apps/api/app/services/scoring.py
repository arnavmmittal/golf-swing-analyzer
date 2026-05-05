"""Score user metrics against tour-level benchmarks.

Each metric receives a score 0-100:
    100  = within ideal range
    50   = at the edge of acceptable range
    0    = outside acceptable range

We also identify the top 3 faults (largest distance from ideal) which become
the focus of the AI coaching feedback.

Each metric has a list of camera views it can be reliably measured from.
Scoring filters by the view of the supplied footage so we don't grade
metrics we couldn't actually measure (e.g., shaft lean from down-the-line).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from app.services.metrics import Metrics

View = Literal["dtl", "face-on"]

BENCHMARKS_PATH = Path(__file__).resolve().parents[4] / "packages" / "benchmarks" / "tour_benchmarks.json"


@dataclass
class MetricScore:
    name: str
    label: str
    value: float
    score: int
    direction: str
    ideal_range: tuple[float, float]
    acceptable_range: tuple[float, float]
    fault: str | None  # "low", "high", or None when in ideal range

    def fault_description(self, benchmark: dict[str, Any]) -> str | None:
        if self.fault == "low":
            return benchmark.get("fault_low") or benchmark.get("fault_high")
        if self.fault == "high":
            return benchmark.get("fault_high") or benchmark.get("fault_low")
        return None


def load_benchmarks(path: Path | None = None) -> dict:
    p = path or BENCHMARKS_PATH
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def score_metrics(
    metrics: Metrics,
    benchmarks: dict | None = None,
    view: View | None = None,
) -> list[MetricScore]:
    """Score metrics against benchmarks.

    If `view` is supplied, only metrics whose `valid_views` includes that
    view are scored — others are dropped from the result.
    """
    benchmarks = benchmarks or load_benchmarks()
    bench_metrics = benchmarks["metrics"]
    values = metrics.as_dict()

    scores: list[MetricScore] = []
    for name, value in values.items():
        b = bench_metrics.get(name)
        if not b:
            continue
        if view is not None:
            valid_views = b.get("valid_views", [])
            if valid_views and view not in valid_views:
                continue
        score, fault = _score_one(value, b)
        scores.append(
            MetricScore(
                name=name,
                label=b["label"],
                value=float(value),
                score=score,
                direction=b["direction"],
                ideal_range=(b["ideal_min"], b["ideal_max"]),
                acceptable_range=(b["acceptable_min"], b["acceptable_max"]),
                fault=fault,
            )
        )
    return scores


def metrics_not_measurable_from_view(
    view: View, benchmarks: dict | None = None
) -> list[dict]:
    """Return metric metadata for metrics this view CAN'T measure reliably.

    Useful for the UI to show a 'needs face-on view' note instead of grading
    the user 0/100 on shaft lean from a DTL clip.
    """
    benchmarks = benchmarks or load_benchmarks()
    bench_metrics = benchmarks["metrics"]
    out = []
    for name, b in bench_metrics.items():
        valid_views = b.get("valid_views", [])
        if valid_views and view not in valid_views:
            out.append({
                "name": name,
                "label": b["label"],
                "valid_views": valid_views,
            })
    return out


def overall_score(scores: list[MetricScore]) -> int:
    """Weighted overall score 0-100. We weight the high-impact metrics more."""
    weights = {
        "tempo_ratio": 1.5,
        "x_factor": 2.0,
        "hip_open_impact": 1.5,
        "spine_stability": 1.0,
        "lead_arm_top": 1.0,
        "head_sway": 1.5,
        "weight_transfer": 1.5,
        "shaft_lean_impact": 1.5,
    }
    if not scores:
        return 0
    total_w = 0.0
    weighted = 0.0
    for s in scores:
        w = weights.get(s.name, 1.0)
        weighted += s.score * w
        total_w += w
    return int(round(weighted / total_w))


def top_faults(scores: list[MetricScore], n: int = 3) -> list[MetricScore]:
    """Return the n metrics with the lowest scores (biggest issues)."""
    faults = [s for s in scores if s.score < 100 and s.fault is not None]
    faults.sort(key=lambda s: s.score)
    return faults[:n]


def _score_one(value: float, benchmark: dict) -> tuple[int, str | None]:
    ideal_min = benchmark["ideal_min"]
    ideal_max = benchmark["ideal_max"]
    acc_min = benchmark["acceptable_min"]
    acc_max = benchmark["acceptable_max"]

    if ideal_min <= value <= ideal_max:
        return 100, None

    if value < ideal_min:
        if value <= acc_min:
            return 0, "low"
        # Linear ramp from 50 (at acc_min) to 100 (at ideal_min).
        ratio = (value - acc_min) / max(ideal_min - acc_min, 1e-9)
        return int(round(50 + 50 * ratio)), "low"

    # value > ideal_max
    if value >= acc_max:
        return 0, "high"
    ratio = (acc_max - value) / max(acc_max - ideal_max, 1e-9)
    return int(round(50 + 50 * ratio)), "high"
