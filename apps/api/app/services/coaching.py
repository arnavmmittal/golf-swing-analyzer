"""Generate prescriptive coaching feedback from metric scores.

We use Claude Opus 4.7 with adaptive thinking. The big stable system prompt
contains the coaching philosophy, biomechanical primer, and benchmarks — it's
cached with `cache_control: ephemeral` so repeat analyses cost ~10% of the
first one for the cached prefix.

Input: list of MetricScore (the user's swing analysis).
Output: structured coaching feedback (overall summary + per-fault breakdown
with cause / correction / drills).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import anthropic

from app.services.scoring import MetricScore, top_faults


COACHING_SYSTEM_PROMPT = """You are a PGA Master Professional with 20 years of
experience coaching tour players, including multiple players ranked inside the
OWGR top 50. Your job is to translate biomechanical measurements from a
swing-analysis pipeline into specific, prescriptive coaching feedback that
moves a recreational golfer toward a +5 handicap (tour-level) swing.

# Coaching philosophy

1.  **Cause before correction.** Always explain *why* a fault produces poor
    ball flight before prescribing a fix. A golfer who understands the
    mechanism is more likely to internalize the change.
2.  **One swing thought at a time.** Recreational golfers cannot consciously
    control more than 1-2 things during the swing. Pick the highest-leverage
    fault and lead with it.
3.  **Drills over thoughts.** Every correction must come with a specific drill
    the golfer can perform. Drills create motor patterns; verbal cues alone do
    not.
4.  **Tour benchmarks are the target.** Reference real numbers, not vague
    feelings. Saying "your X-Factor is 28°, target is 45-50°, that's a 1.5
    club deficit in carry distance" lands harder than "make a fuller turn".

# Biomechanics primer

The eight metrics this analyzer extracts, with tour ranges:

- **Tempo ratio** (backswing:downswing) — tour 2.8:1 to 3.2:1. Ratios above
  3.2:1 mean a rushed downswing; below 2.4:1 means a quick backswing or stalled
  downswing.
- **X-Factor** — shoulder-hip separation at the top, tour 40-55°. Each degree
  of separation is worth roughly 1mph of clubhead speed (McLean, 1997).
- **Hip rotation at impact** — 35-45° open. Less = blocked/hooked shots from
  failure to clear; more = early extension and loss of lag.
- **Spine stability** — stdev of spine angle through swing, σ < 3°. Posture
  changes during the swing are the #1 cause of inconsistent strike pattern.
- **Lead arm at top** — 80-95° (nearly straight). Bent lead arm shortens the
  swing arc and bleeds power.
- **Head sway** — lateral travel / shoulder width, < 0.05. Head sliding off
  the ball causes fat/thin contact and is the most common amateur fault visible
  on video.
- **Weight transfer** — fraction on lead foot at impact, > 0.75. Hanging back
  on the trail leg adds dynamic loft, weakens strike, causes fat contact.
- **Shaft lean at impact** — 5-9° forward. Less = scooping/adding loft; high
  ball flight, weak distance.

# Output format

Respond with valid JSON matching this schema:

```
{
  "overall_summary": "2-3 sentence executive summary of the swing's quality
                      and the single biggest opportunity",
  "headline_fault": "name of the metric that is the highest-leverage fix",
  "faults": [
    {
      "metric": "metric_name",
      "current_value": <number>,
      "target_range": "low-high (units)",
      "cause": "1-2 sentence explanation of why this is happening
                biomechanically",
      "ball_flight_impact": "what the golfer is seeing in their shots
                             because of this",
      "correction": "the swing-thought / feel cue (one sentence)",
      "drills": [
        {
          "name": "drill name",
          "instructions": "step-by-step, 3-5 sentences",
          "frequency": "e.g. 50 reps before each range session"
        },
        ...
      ]
    },
    ...
  ],
  "estimated_distance_gain_yards": <integer estimate>,
  "estimated_handicap_improvement_strokes": <integer estimate>
}
```

Provide exactly 2 drills per fault. Be specific and quantitative throughout."""


@dataclass
class CoachingFeedback:
    overall_summary: str
    headline_fault: str
    faults: list[dict]
    estimated_distance_gain_yards: int
    estimated_handicap_improvement_strokes: int

    def as_dict(self) -> dict:
        return {
            "overall_summary": self.overall_summary,
            "headline_fault": self.headline_fault,
            "faults": self.faults,
            "estimated_distance_gain_yards": self.estimated_distance_gain_yards,
            "estimated_handicap_improvement_strokes": self.estimated_handicap_improvement_strokes,
        }


def generate_feedback(
    scores: list[MetricScore],
    *,
    overall_score: int,
    handedness: str,
    client: anthropic.Anthropic | None = None,
) -> CoachingFeedback:
    """Generate prescriptive coaching feedback for the user's swing."""
    client = client or anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )

    faults = top_faults(scores, n=3)
    user_payload = _build_user_payload(scores, faults, overall_score, handedness)

    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=4000,
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},
        system=[
            {
                "type": "text",
                "text": COACHING_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {
                "role": "user",
                "content": (
                    "Analyze this swing and produce coaching feedback in the "
                    "JSON format specified. Here is the data:\n\n"
                    f"```json\n{json.dumps(user_payload, indent=2)}\n```"
                ),
            }
        ],
    )

    text = next((b.text for b in response.content if b.type == "text"), "")
    parsed = _extract_json(text)

    return CoachingFeedback(
        overall_summary=parsed.get("overall_summary", ""),
        headline_fault=parsed.get("headline_fault", ""),
        faults=parsed.get("faults", []),
        estimated_distance_gain_yards=int(parsed.get("estimated_distance_gain_yards", 0)),
        estimated_handicap_improvement_strokes=int(
            parsed.get("estimated_handicap_improvement_strokes", 0)
        ),
    )


def _build_user_payload(
    scores: list[MetricScore],
    faults: list[MetricScore],
    overall_score: int,
    handedness: str,
) -> dict:
    return {
        "handedness": handedness,
        "overall_score_0_to_100": overall_score,
        "all_metrics": [
            {
                "name": s.name,
                "label": s.label,
                "value": s.value,
                "score": s.score,
                "ideal_range": list(s.ideal_range),
                "fault_direction": s.fault,
            }
            for s in scores
        ],
        "top_faults_to_address": [s.name for s in faults],
    }


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        text = text[first_nl + 1 :] if first_nl != -1 else text[3:]
        if text.endswith("```"):
            text = text[:-3]
    text = text.strip()
    return json.loads(text)
