# Operations Guide

This project is now organized around a serving manifest. The manifest is the
deployment contract: it names the backbone, router, adapters, quality gates, and
runtime metadata for one release.

## 1. Validate Local Artifacts

Run this before starting the API:

```bash
python scripts/validate_artifacts.py --manifest configs/serving_manifest.json
```

Expected result:

```json
{
  "ok": true,
  "missing": []
}
```

If this fails, the model may still run through fallback code, but the declared
release is incomplete.

## 2. Run A Release Quality Gate

Prefetch the router encoders and Qwen backbone before the first full evaluation
or before starting the API in a fresh environment:

```bash
python scripts/prefetch_models.py --manifest configs/serving_manifest.json
```

The first run can be slow because it downloads Qwen2.5-VL checkpoint shards.
Later runs are faster as long as the HuggingFace cache is preserved.
After prefetching, set `HF_LOCAL_FILES_ONLY=1` to force serving/evaluation to
use cache only and avoid startup calls to HuggingFace.

Generate routed predictions on a validation set:

```bash
python scripts/evaluate_routed.py \
  --manifest configs/serving_manifest.json \
  --metadata-path data/processed/multitask/validation.jsonl \
  --predictions-path outputs/predictions/routed_validation.jsonl
```

For a quick smoke run, add `--limit 3`.
To verify that the backbone and adapters can be loaded without waiting for
generation on a slow CPU/MPS machine:

```bash
HF_LOCAL_FILES_ONLY=1 python scripts/evaluate_routed.py \
  --manifest configs/serving_manifest.json \
  --metadata-path data/processed/multitask/validation.jsonl \
  --load-only
```

Then gate the release:

```bash
python scripts/run_release_gate.py \
  --manifest configs/serving_manifest.json \
  --predictions outputs/predictions/routed_validation.jsonl \
  --report-out outputs/reports/release_gate_report.json
```

The predictions JSONL must include:

```json
{
  "task_type": "chartqa",
  "question": "What is the value?",
  "answers": ["42"],
  "prediction": "42",
  "predicted_task_type": "chartqa"
}
```

`predicted_task_type` is optional. If it is present for every row, the gate also
checks router accuracy.

Current primary quality gates:

- ChartQA: `chart_hybrid_accuracy >= 0.80`
- DocVQA: `docvqa_anls >= 0.84`
- TextVQA: `textvqa_vqa_score >= 0.74`
- Router: `routing_accuracy >= 0.98`

## 3. Start The API

```bash
ROUTED_VLM_MANIFEST=configs/serving_manifest.json \
INFERENCE_LOG_PATH=outputs/logs/inference.jsonl \
HF_LOCAL_FILES_ONLY=1 \
python scripts/serve_api.py
```

Useful endpoints:

- `GET /health`: readiness status
- `GET /metadata`: active manifest, model, router, and loaded adapters
- `GET /metrics`: process-local request counters and average latency
- `POST /predict`: image-question inference

## 4. Local Mac MPS Profile

On Apple Silicon, local GPU acceleration uses Apple MPS/Metal. It is not CUDA.
Use it for local smoke checks and development, not for full release evaluation
of the 7B model.

Check the accelerator:

```bash
make check-accelerator
```

Fresh Mac environment:

```bash
conda create -n routed-vlm-mps python=3.12 -y
conda activate routed-vlm-mps
pip install -r requirements-mac.txt
pip install -r requirements.txt
make prefetch-models
```

Load-only smoke test:

```bash
HF_LOCAL_FILES_ONLY=1 ROUTED_VLM_DEVICE=mps \
python scripts/evaluate_routed.py \
  --manifest configs/serving_manifest.json \
  --metadata-path data/processed/multitask/validation.jsonl \
  --load-only
```

If MPS is unavailable, recreate the environment with the official macOS PyTorch
wheels and rerun `make check-accelerator`.

## 5. Inspect Inference Logs

Each prediction appends one JSONL record to `INFERENCE_LOG_PATH`.

Logged fields include:

- `request_id`
- `manifest_name`
- `model_name`
- `question`
- image filename, content type, byte size, and SHA-256
- router decision
- answer
- latency
- error, when a request fails

The log intentionally stores an image hash and temp path metadata, not image
bytes.

## 6. Docker Runtime

Build:

```bash
docker build -t routed-vlm-qa .
```

Run:

```bash
docker run --rm -p 8000:8000 \
  --env-file .env.example \
  -v "$PWD/checkpoints:/app/checkpoints" \
  -v "$PWD/outputs:/app/outputs" \
  -v "$PWD/.cache/huggingface:/app/.cache/huggingface" \
  routed-vlm-qa
```

For GPU deployment, use an NVIDIA runtime image or a CUDA-enabled PyTorch base
image and mount the same `checkpoints/` and `outputs/` directories.
Also mount a persistent HuggingFace cache; otherwise every new container may
download the 7B backbone again.

For a CUDA runtime, use:

```bash
ROUTED_VLM_DEVICE=cuda
ROUTED_VLM_LOAD_IN_4BIT=1
HF_LOCAL_FILES_ONLY=1
```

`ROUTED_VLM_LOAD_IN_4BIT=1` is only valid on CUDA. It is the recommended path
for serving the 7B backbone on a constrained GPU. On CPU/MPS, expect 7B
generation to be very slow; use `--load-only` for local smoke checks and run
full evaluation on a GPU machine.

## 7. GPU Notebook/Lab Profile

The 7B backbone is best served from a CUDA GPU notebook/lab box. Recommended:

```bash
pip install -r requirements-cuda.txt --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
make validate-artifacts
make prefetch-models
HF_LOCAL_FILES_ONLY=1 ROUTED_VLM_DEVICE=cuda ROUTED_VLM_LOAD_IN_4BIT=1 \
  python scripts/serve_api.py
```

Run full `make evaluate-routed` and `make release-gate` on this GPU profile.
Local Mac MPS is expected to be much slower for real generation.

## 8. Release Checklist

Before calling a release production-ready:

1. Copy router and adapter artifacts into `checkpoints/`.
2. Run `scripts/validate_artifacts.py`.
3. Run `scripts/prefetch_models.py`.
4. Generate validation predictions for the manifest.
5. Run `scripts/run_release_gate.py`.
6. Start the API with `ROUTED_VLM_MANIFEST`.
7. Check `/metadata`.
8. Send a smoke `/predict` request for each task type.
9. Check `/metrics` and `outputs/logs/inference.jsonl`.

## Version Notes

The current router classifier was saved with `scikit-learn==1.6.1`, so the
runtime dependency is pinned to that version to avoid joblib compatibility drift.
