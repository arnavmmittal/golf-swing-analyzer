# Golf Swing Analyzer — Design Document

**Date**: 2026-05-04
**Goal**: Take a video of any golf swing, output specific, actionable feedback that pushes the user toward a +5 handicap (tour-level) swing.

## Why this works

A "+5 handicap swing" is not subjective. Tour-level swings cluster tightly around measurable biomechanical targets:

| Metric | Tour Range | Why it matters |
|---|---|---|
| Tempo ratio (back:down) | 2.8 : 1 to 3.2 : 1 | Sequencing, rhythm |
| X-Factor at top | 40°–55° | Power generation |
| Hip rotation at impact | 35°–45° open | Lead-side clearance |
| Spine angle stability | σ < 3° | Consistent strike |
| Lead arm angle at top | 80°–95° | Width, plane |
| Head sway | < 2 inches lateral | Steady center |
| Weight transfer | 80%+ to lead at impact | Ball-first contact |
| Shaft lean at impact | 5°–8° forward | Compression |

If we can measure these, we can compute deltas, flag the biggest faults, and prescribe drills.

## Architecture

```
Browser ──► Next.js (Vercel) ──► Python FastAPI ──► MediaPipe pose
                                       │
                                       ▼
                                  Metrics JSON
                                       │
                                       ▼
                                  Claude Opus 4.7
                                       │
                                       ▼
                              Feedback + Drills + Annotated MP4
```

**Why split frontend from backend**: MediaPipe + OpenCV are heavyweight Python deps that don't fit the Vercel serverless runtime cleanly. Keeping the inference service separate lets us scale it independently and swap in a GPU host later.

## Pipeline

1. **Ingest** — accept mp4/mov up to 60s, normalize to 60fps, store in Vercel Blob
2. **Pose** — MediaPipe BlazePose (full mode, 33 3D landmarks per frame)
3. **Phase segment** — detect Address / Takeaway / Top / Downswing / Impact / Finish via wrist velocity zero-crossings and peaks
4. **Metric extraction** — compute the eight metrics above per frame, then aggregate per phase
5. **Benchmark diff** — score each metric 0–100 vs tour ranges; flag the three largest deltas
6. **AI coaching** — Claude turns the metric deltas into prescriptive feedback (cause → correction → drill)
7. **Render** — OpenCV writes an annotated MP4: skeleton overlay, phase markers, swing-plane line, key angle readouts
8. **Return** — JSON metrics + annotated video URL + feedback markdown

## Tech stack

- **Frontend**: Next.js 16 (App Router), TypeScript, Tailwind, shadcn/ui
- **Backend**: Python 3.11, FastAPI, MediaPipe, OpenCV, NumPy, SciPy
- **AI**: `claude-opus-4-7` via `anthropic` SDK with prompt caching on the benchmark/coaching system prompt
- **Storage**: Vercel Blob for videos
- **Deploy**: Vercel for web, Modal/Render for the Python API

## Repo layout

```
golf-swing-analyzer/
├── apps/
│   ├── web/           # Next.js
│   └── api/           # FastAPI
├── packages/
│   └── benchmarks/    # tour-level metric ranges (JSON)
├── docs/plans/        # this file
└── README.md
```

## Build sequence (commit-by-commit)

1. Scaffold + design doc + license + README skeleton
2. Python: pose extraction service (one frame in, landmarks out)
3. Python: phase segmentation
4. Python: metric extraction with tour benchmarks
5. Python: Claude coaching integration
6. Python: annotated video renderer
7. Python: FastAPI endpoint wiring it all together
8. Web: upload UI + video preview
9. Web: results dashboard with feedback cards
10. Deploy configs (Vercel + Modal)
11. README + sample video

## What we're explicitly not building

- Mobile app (web works on phones)
- Multiple-angle stitching (one camera angle at a time)
- Real-time analysis (offline only)
- User accounts / swing history (single-shot tool first)
- Club-head detection (we infer from hand path; full detection is a v2)
