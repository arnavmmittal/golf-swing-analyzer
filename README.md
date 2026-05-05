# Golf Swing Analyzer

Upload a swing video. Get specific, biomechanics-grounded feedback that pushes you toward a tour-level (+5 handicap) swing.

## How it works

1. **Pose extraction** — MediaPipe BlazePose tracks 33 body landmarks per frame
2. **Phase segmentation** — detects Address, Takeaway, Top, Impact, and Finish from wrist velocity
3. **Metric extraction** — computes eight biomechanical metrics (tempo, X-Factor, hip rotation, spine stability, head sway, etc.)
4. **Benchmark comparison** — scores each metric against tour-level ranges
5. **AI coaching** — Claude Opus 4.7 turns the deltas into prescriptive corrections and drills
6. **Annotated video** — skeleton overlay, phase markers, swing-plane line

## Project layout

```
apps/
├── web/           Next.js frontend
└── api/           FastAPI inference service
packages/
└── benchmarks/    Tour-level metric ranges
docs/plans/        Design docs
```

## Quick start

```bash
# Backend
cd apps/api
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload

# Frontend
cd apps/web
npm install
npm run dev
```

Set `ANTHROPIC_API_KEY` in `apps/api/.env`.

## Status

Active development. See [docs/plans/2026-05-04-golf-swing-analyzer-design.md](docs/plans/2026-05-04-golf-swing-analyzer-design.md) for the full design.

## License

MIT
