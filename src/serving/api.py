"""FastAPI app for routed VLM QA inference."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
import os

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from src.ops.inference_logging import (
    InferenceJsonlLogger,
    InferenceStats,
    UploadedImageInfo,
    dataclass_to_dict,
    image_info_from_upload,
    new_request_id,
)
from src.routing.task_router import (
    DEFAULT_MIN_CONFIDENCE,
)
from src.serving.config import build_service_from_environment


app = FastAPI(
    title="Routed VLM QA API",
    version="0.1.0",
    description="Route ChartQA, DocVQA, and TextVQA questions to the best backend.",
)


class PredictResponse(BaseModel):
    """JSON response returned by the /predict endpoint."""

    request_id: str
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


class MetadataResponse(BaseModel):
    """Deployment metadata returned by the /metadata endpoint."""

    manifest: dict | None
    model_name: str | None
    router_dir: str | None
    loaded_adapters: list[str]
    require_adapters: bool | None


class MetricsResponse(BaseModel):
    """Process-local serving metrics for lightweight monitoring."""

    total_requests: int
    total_errors: int
    success_count: int
    avg_latency_seconds: float | None
    task_counts: dict[str, int]
    backend_counts: dict[str, int]
    adapter_counts: dict[str, int]


@app.on_event("startup")
def startup() -> None:
    """Load long-lived server objects once when the API starts."""
    service = build_service_from_environment()
    app.state.service = service.load()
    app.state.model_loaded = True
    app.state.inference_logger = InferenceJsonlLogger(
        os.getenv("INFERENCE_LOG_PATH", "outputs/logs/inference.jsonl")
    )
    app.state.inference_stats = InferenceStats()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return API readiness information."""
    service = getattr(app.state, "service", None)
    return HealthResponse(
        status="ok",
        router_loaded=service is not None and service.router is not None,
        model_loaded=bool(app.state.model_loaded),
    )


@app.get("/metadata", response_model=MetadataResponse)
def metadata() -> MetadataResponse:
    """Return operational metadata for the loaded service."""
    service = getattr(app.state, "service", None)
    if service is None:
        return MetadataResponse(
            manifest=None,
            model_name=None,
            router_dir=None,
            loaded_adapters=[],
            require_adapters=None,
        )

    manifest = service.manifest.to_metadata() if service.manifest else None
    return MetadataResponse(
        manifest=manifest,
        model_name=service.model_name,
        router_dir=str(service.router_dir),
        loaded_adapters=sorted(service.loaded_adapters),
        require_adapters=service.require_adapters,
    )


@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    """Return process-local inference counters."""
    stats = getattr(app.state, "inference_stats", None)
    if stats is None:
        stats = InferenceStats()
    return MetricsResponse(**stats.snapshot())


async def save_upload_to_temp_file(image: UploadFile) -> UploadedImageInfo:
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
        image_path = handle.name
    return image_info_from_upload(
        path=image_path,
        filename=image.filename,
        content_type=image.content_type,
        content=content,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(
    question: str = Form(...),
    image: UploadFile = File(...),
    min_confidence: float = Form(DEFAULT_MIN_CONFIDENCE),
) -> PredictResponse:
    """Generate an answer for one image-question pair."""
    started_at = time.perf_counter()
    request_id = new_request_id()

    clean_question = question.strip()
    if not clean_question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    image_info = await save_upload_to_temp_file(image)
    service = getattr(app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Inference service is not loaded.")

    try:
        routed_prediction = service.predict(
            image_path=image_info.path,
            question=clean_question,
            min_confidence=min_confidence,
        )
    except Exception as exc:
        stats = getattr(app.state, "inference_stats", None)
        if stats is not None:
            stats.record_error()
        log_inference_event(
            request_id=request_id,
            question=clean_question,
            image_info=image_info,
            latency_seconds=time.perf_counter() - started_at,
            error=str(exc),
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    decision = routed_prediction.decision
    latency = time.perf_counter() - started_at
    stats = getattr(app.state, "inference_stats", None)
    if stats is not None:
        stats.record_success(
            task_type=decision.task_type,
            backend=decision.backend_name,
            adapter=decision.adapter_name,
            latency_seconds=latency,
        )
    log_inference_event(
        request_id=request_id,
        question=clean_question,
        image_info=image_info,
        latency_seconds=latency,
        answer=routed_prediction.answer,
        decision={
            "task_type": decision.task_type,
            "backend": decision.backend_name,
            "use_adapter": decision.use_adapter,
            "adapter": decision.adapter_name,
            "confidence": decision.confidence,
        },
    )

    return PredictResponse(
        request_id=request_id,
        answer=routed_prediction.answer,
        question=clean_question,
        task_type=decision.task_type,
        backend=decision.backend_name,
        use_adapter=decision.use_adapter,
        adapter=decision.adapter_name,
        confidence=decision.confidence,
        latency_seconds=latency,
    )


def log_inference_event(
    request_id: str,
    question: str,
    image_info: UploadedImageInfo,
    latency_seconds: float,
    answer: str | None = None,
    decision: dict | None = None,
    error: str | None = None,
) -> None:
    """Append one inference event if the runtime logger is configured."""
    logger = getattr(app.state, "inference_logger", None)
    if logger is None:
        return
    service = getattr(app.state, "service", None)
    logger.log(
        {
            "request_id": request_id,
            "manifest_name": service.manifest.name if service and service.manifest else None,
            "model_name": service.model_name if service else None,
            "question": question,
            "image": dataclass_to_dict(image_info),
            "latency_seconds": latency_seconds,
            "answer": answer,
            "decision": decision,
            "error": error,
        }
    )
