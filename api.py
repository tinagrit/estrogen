from __future__ import annotations

import csv
from io import StringIO
import json
from pathlib import Path
import re
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from models import ACTIVITY_PATH, METRICS_PATH, TOXICITY_PATH
from service import (
    generate_candidates,
    molecule_png,
    optimize_candidates,
    score_molecule,
    sdf_export,
)


WEB_DIR = Path(__file__).resolve().parent / "web"


Objective = Literal[
    "Decrease LogP",
    "Lower Predicted ClinTox Toxicity",
    "Increase Predicted ERα Activity",
]


class GenerateRequest(BaseModel):
    prefix: str = Field(min_length=1, max_length=500)
    batch_size: int = Field(default=20, ge=1, le=50)
    temperature: float = Field(default=0.8, ge=0.05, le=2.0)
    top_k: int = Field(default=50, ge=1, le=200)
    max_length: int = Field(default=80, ge=1, le=200)


class OptimizeRequest(BaseModel):
    smiles: str = Field(min_length=1, max_length=2000)
    objective: Objective
    limit: int = Field(default=15, ge=1, le=50)


class ScoreRequest(BaseModel):
    smiles: str = Field(min_length=1, max_length=2000)


class ExportRequest(BaseModel):
    rows: list[dict[str, Any]] = Field(max_length=500)
    filename: str = Field(default="molecule_candidates", min_length=1, max_length=80)


app = FastAPI(
    title="ERalpha Molecule Studio API",
    version="1.0.0",
    description=(
        "Computational molecule generation, filtering, activity scoring, toxicity "
        "scoring, and lead-optimization endpoints."
    ),
)


def _safe_filename(filename: str, extension: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", filename).strip("_")
    return f"{stem or 'molecule_candidates'}.{extension}"


def _download_response(content: bytes, media_type: str, filename: str) -> Response:
    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/models")
def model_status() -> dict[str, Any]:
    activity_available = ACTIVITY_PATH.is_file()
    toxicity_available = TOXICITY_PATH.is_file()
    return {
        "activity": {
            "available": activity_available,
            "display_name": "Predicted ERalpha activity probability",
            "artifact": ACTIVITY_PATH.name,
        },
        "toxicity": {
            "available": toxicity_available,
            "display_name": "Predicted ClinTox toxicity probability",
            "artifact": TOXICITY_PATH.name,
        },
        "evaluation_available": METRICS_PATH.is_file(),
        "using_placeholder": not (activity_available and toxicity_available),
    }


@app.get("/api/metrics")
def model_metrics() -> dict[str, Any]:
    if not METRICS_PATH.is_file():
        return {
            "available": False,
            "message": "Run python3 train_models.py to generate held-out metrics.",
        }
    try:
        report = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=500, detail="The metrics report is unreadable.") from error
    return {"available": True, **report}


@app.post("/api/generate")
async def generate(request: GenerateRequest) -> dict[str, Any]:
    try:
        return await run_in_threadpool(
            generate_candidates,
            request.prefix,
            request.batch_size,
            request.temperature,
            request.top_k,
            request.max_length,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/api/optimize")
async def optimize(request: OptimizeRequest) -> dict[str, Any]:
    try:
        return await run_in_threadpool(
            optimize_candidates,
            request.smiles,
            request.objective,
            request.limit,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/api/score")
async def score(request: ScoreRequest) -> dict[str, str | float | int]:
    result = await run_in_threadpool(score_molecule, request.smiles.strip())
    if result is None:
        raise HTTPException(status_code=400, detail="Enter a complete, valid SMILES string.")
    return result


@app.get("/api/molecule-image")
async def molecule_image(
    smiles: str = Query(min_length=1, max_length=2000),
) -> Response:
    try:
        content = await run_in_threadpool(molecule_png, smiles)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return Response(content=content, media_type="image/png")


@app.post("/api/export/sdf")
async def export_sdf(request: ExportRequest) -> Response:
    content = await run_in_threadpool(sdf_export, request.rows)
    return _download_response(
        content.encode("utf-8"),
        "chemical/x-mdl-sdfile",
        _safe_filename(request.filename, "sdf"),
    )


@app.post("/api/export/csv")
def export_csv(request: ExportRequest) -> Response:
    fieldnames = list(dict.fromkeys(key for row in request.rows for key in row))
    output = StringIO()
    if fieldnames:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(request.rows)
    return _download_response(
        output.getvalue().encode("utf-8"),
        "text/csv; charset=utf-8",
        _safe_filename(request.filename, "csv"),
    )


# This makes the Python process useful as a one-command development server. In the
# deployed two-process setup, Axum serves the same directory and proxies only /api/*.
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
