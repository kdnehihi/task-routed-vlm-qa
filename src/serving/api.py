"""FastAPI app for routed VLM QA inference."""

from __future__ import annotations

import tempfile
import time
import os
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from src.routing.task_router import (
    DEFAULT_MIN_CONFIDENCE,
)
from src.serving.routed_vlm import RoutedVLMService


app = FastAPI(
    title="Routed VLM QA API",
    version="0.1.0",
    description="Route ChartQA, DocVQA, and TextVQA questions to the best backend.",
)


class PredictResponse(BaseModel):
    """JSON response returned by the /predict endpoint."""

    answer: str
    question: str
    task_type: str
    backend: str
    use_adapter: bool
    adapter: str | None = None
    confidence: float | None = None
    latency_seconds: float


class HealthResponse(BaseModel):
    """JSON response returned by the /health endpoint."""

    status: str
    router_loaded: bool
    model_loaded: bool


@app.on_event("startup")
def startup() -> None:
    """Load long-lived server objects once when the API starts."""
    require_adapters = os.getenv("REQUIRE_ADAPTERS", "1") not in {"0", "false", "False"}
    app.state.service = RoutedVLMService(require_adapters=require_adapters).load()
    app.state.model_loaded = True


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return API readiness information."""
    service = getattr(app.state, "service", None)
    return HealthResponse(
        status="ok",
        router_loaded=service is not None and service.router is not None,
        model_loaded=bool(app.state.model_loaded),
    )


async def save_upload_to_temp_file(image: UploadFile) -> str:
    """Save an uploaded image to a temporary local path."""
    suffix = Path(image.filename or "image.png").suffix or ".png"

    try:
        content = await image.read()
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Could not read uploaded image.",
        ) from exc

    if not content:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(content)
        return handle.name


@app.post("/predict", response_model=PredictResponse)
async def predict(
    question: str = Form(...),
    image: UploadFile = File(...),
    min_confidence: float = Form(DEFAULT_MIN_CONFIDENCE),
) -> PredictResponse:
    """Generate an answer for one image-question pair."""
    started_at = time.perf_counter()

    clean_question = question.strip()
    if not clean_question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    image_path = await save_upload_to_temp_file(image)
    service = getattr(app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Inference service is not loaded.")

    try:
        routed_prediction = service.predict(
            image_path=image_path,
            question=clean_question,
            min_confidence=min_confidence,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    decision = routed_prediction.decision
    latency = time.perf_counter() - started_at

    return PredictResponse(
        answer=routed_prediction.answer,
        question=clean_question,
        task_type=decision.task_type,
        backend=decision.backend_name,
        use_adapter=decision.use_adapter,
        adapter=decision.adapter_name,
        confidence=decision.confidence,
        latency_seconds=latency,
    )
