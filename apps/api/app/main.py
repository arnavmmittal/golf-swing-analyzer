"""FastAPI service that runs the full swing-analysis pipeline.

POST /analyze accepts a video upload, runs:
    pose extraction → phase detection → metrics → scoring → AI coaching →
    annotated video render

and returns JSON with all results plus a URL to the annotated video.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import uuid
from pathlib import Path

from dotenv import load_dotenv
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from app.services.coaching import generate_feedback
from app.services.metrics import compute_metrics
from app.services.phases import detect_phases
from app.services.pose import extract_pose, stack_landmarks, stack_world_landmarks
from app.services.render import render_annotated
from app.services.scoring import overall_score, score_metrics

load_dotenv()
logger = logging.getLogger("golf_swing_analyzer")
logging.basicConfig(level=logging.INFO)

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/tmp/golf-analyzer-output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi"}
MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200MB

app = FastAPI(title="Golf Swing Analyzer", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(
    video: UploadFile = File(...),
    handedness: Literal["right", "left", "auto"] = Form("auto"),
) -> JSONResponse:
    suffix = Path(video.filename or "upload.mp4").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    analysis_id = uuid.uuid4().hex
    work_dir = Path(tempfile.mkdtemp(prefix=f"swing-{analysis_id}-"))
    input_path = work_dir / f"input{suffix}"
    output_path = OUTPUT_DIR / f"{analysis_id}.mp4"

    try:
        await _save_upload(video, input_path)

        logger.info("Extracting pose for %s", analysis_id)
        frames, meta = extract_pose(input_path)
        if len(frames) < 20:
            raise HTTPException(
                status_code=400,
                detail="Video too short or no pose detected (need >20 frames).",
            )

        landmarks = stack_landmarks(frames)
        world_landmarks = stack_world_landmarks(frames)
        logger.info("Detecting phases for %s (handedness=%s)", analysis_id, handedness)
        override = None if handedness == "auto" else handedness
        phases = detect_phases(
            landmarks,
            fps=meta["fps"],
            handedness_override=override,
            world_landmarks=world_landmarks,
        )

        logger.info("Computing metrics for %s", analysis_id)
        metrics = compute_metrics(world_landmarks, phases, fps=meta["fps"])
        scores = score_metrics(metrics)
        overall = overall_score(scores)

        logger.info("Generating coaching feedback for %s", analysis_id)
        feedback = generate_feedback(
            scores,
            overall_score=overall,
            handedness=phases.handedness,
        )

        logger.info("Rendering annotated video for %s", analysis_id)
        render_annotated(
            input_path, output_path, landmarks, phases, fps=meta["fps"]
        )

        result_payload = {
                "analysis_id": analysis_id,
                "video_meta": meta,
                "phases": phases.as_dict(),
                "metrics": metrics.as_dict(),
                "scores": [
                    {
                        "name": s.name,
                        "label": s.label,
                        "value": s.value,
                        "score": s.score,
                        "ideal_range": list(s.ideal_range),
                        "fault": s.fault,
                    }
                    for s in scores
                ],
                "overall_score": overall,
                "feedback": feedback.as_dict(),
                "annotated_video_url": f"/video/{analysis_id}",
            }

        # Persist for debugging/inspection alongside the rendered video.
        result_path = OUTPUT_DIR / f"{analysis_id}.json"
        result_path.write_text(_json_dumps(result_payload))

        return JSONResponse(result_payload)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


@app.get("/video/{analysis_id}")
def get_video(analysis_id: str) -> FileResponse:
    if not _is_safe_id(analysis_id):
        raise HTTPException(status_code=400, detail="Invalid analysis id")
    path = OUTPUT_DIR / f"{analysis_id}.mp4"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Video not found or expired")
    return FileResponse(path, media_type="video/mp4", filename=f"swing-{analysis_id}.mp4")


async def _save_upload(upload: UploadFile, dest: Path) -> None:
    bytes_written = 0
    with dest.open("wb") as f:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            bytes_written += len(chunk)
            if bytes_written > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="File too large (>200MB)")
            f.write(chunk)


def _is_safe_id(s: str) -> bool:
    return len(s) == 32 and all(c in "0123456789abcdef" for c in s)


def _json_dumps(obj: dict) -> str:
    import json
    return json.dumps(obj, indent=2, default=str)
