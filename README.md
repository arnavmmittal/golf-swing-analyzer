# Golf Swing Analyzer

Upload a swing video. Get specific, biomechanics-grounded feedback that pushes you toward a tour-level (+5 handicap) swing.

[![CI](https://github.com/arnavmmittal/golf-swing-analyzer/actions/workflows/ci.yml/badge.svg)](https://github.com/arnavmmittal/golf-swing-analyzer/actions/workflows/ci.yml)

## Why this works

A "+5 handicap swing" isn't subjective. Tour-level swings cluster tightly around eight measurable biomechanical targets:

| Metric | Tour range | What it tells us |
|---|---|---|
| Tempo ratio (back : down) | 2.8 : 1 to 3.2 : 1 | Rhythm and sequencing |
| X-Factor at top | 40°–55° | Power generation potential |
| Hip rotation at impact | 35°–45° open | Lead-side clearance |
| Spine angle stability | σ < 3° | Strike consistency |
| Lead arm at top | 80°–95° | Width and swing arc |
| Head sway | < 0.05 (norm.) | Steady center |
| Weight transfer | > 75% to lead | Compression, ball-first contact |
| Shaft lean at impact | 5°–9° forward | Loft control, flight quality |

If we can measure these, we can compute deltas, flag the highest-leverage faults, and prescribe specific drills. We measure them. Then we ask Claude Opus 4.7 — instructed as a PGA Master Professional — to translate the deltas into prescriptive coaching with cause, correction, and 2 drills per fault.

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│  Next.js UI │ ──► │ Python FastAPI   │ ──► │ Claude API  │
│  (Vercel)   │     │ MediaPipe+OpenCV │     │ (feedback)  │
└─────────────┘     └──────────────────┘     └─────────────┘
                            │
                            ▼
                    Annotated MP4 + JSON
```

## Pipeline

1. **Pose extraction** — MediaPipe BlazePose (model_complexity=2), 33 3D landmarks per frame
2. **Phase segmentation** — wrist-velocity profile detects Address / Top / Impact / Finish
3. **Metric extraction** — eight biomechanical metrics computed in image space (scale-invariant)
4. **Benchmark scoring** — each metric scored 0–100 vs tour ranges; overall score is weighted average (X-Factor 2×)
5. **AI coaching** — Claude Opus 4.7 (adaptive thinking, prompt caching) turns metric deltas into cause + correction + 2 drills per fault
6. **Annotated render** — OpenCV writes MP4 with skeleton overlay, shoulder/hip/spine axes, phase markers

## Repo layout

```
apps/
├── api/                 # Python FastAPI service
│   ├── app/
│   │   ├── main.py      # /analyze + /video/{id}
│   │   ├── services/    # pose, phases, metrics, scoring, coaching, render
│   │   └── models/      # landmark indices, dataclasses
│   ├── tests/           # geometry + scoring unit tests
│   ├── Dockerfile
│   └── requirements.txt
├── web/                 # Next.js 15 frontend
│   └── src/
│       ├── app/         # page.tsx, layout.tsx
│       ├── components/  # UploadCard, Results, MetricCard, FaultCard
│       └── lib/         # types
packages/
└── benchmarks/          # tour_benchmarks.json — sourced ranges
docs/plans/              # design doc
.github/workflows/ci.yml # pytest + tsc + next build
```

## Quick start

### 1. Backend

```bash
cd apps/api
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # set ANTHROPIC_API_KEY
uvicorn app.main:app --reload --port 8000
```

### 2. Frontend

```bash
cd apps/web
npm install
cp .env.example .env.local
npm run dev  # http://localhost:3000
```

### 3. Run tests

```bash
cd apps/api
pip install pytest
pytest -q
```

### 4. Docker (backend)

From the repo root:

```bash
docker build -t golf-analyzer-api -f apps/api/Dockerfile .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY golf-analyzer-api
```

## Deployment

- **Frontend** → Vercel. Set `NEXT_PUBLIC_API_BASE_URL` to the public URL of the FastAPI service.
- **Backend** → any container host that allows long-lived requests (Render, Railway, Fly.io, Modal). Vercel's serverless runtime can't host MediaPipe + OpenCV cleanly. Set `ANTHROPIC_API_KEY` and `CORS_ORIGINS` (comma-separated allowlist).

## Tips for the best analysis

- **Camera angle**: side-on (down-the-line behind the ball) or face-on. Pick one and keep it constant.
- **Lighting**: outdoors or bright indoors — BlazePose loses accuracy in low light.
- **Frame**: full body in frame from setup through follow-through.
- **Length**: 5–10 seconds is plenty. Longer videos cost more pose-extraction time without adding signal.
- **FPS**: 60fps is the sweet spot. 30fps works; 240fps doesn't add much because BlazePose tops out around 30 inferences per second on CPU.

## Sources

- McLean, J. *The X-Factor Swing*. Harper Collins, 1997.
- Cheetham, P. et al. "A computer-aided method for analyzing the golf swing." *J. Sport Sci.*, 2001.
- Hume, P., Keogh, J., & Reid, D. "A review of full-swing biomechanics." *Sports Med.*, 2005.
- TPI (Titleist Performance Institute) screening data.

## License

MIT — see [LICENSE](LICENSE).
