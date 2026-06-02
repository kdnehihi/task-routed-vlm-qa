# Roadmap

## Phase 0: Repository Scaffold

Status: in progress

- Create professional repository structure.
- Add documentation and placeholder modules.
- Define intended research direction and comparison baselines.
- Avoid implementing real training, routing, or model code.

## Phase 1: Data Contracts

- Define a shared JSONL schema.
- Add validation tests for required fields.
- Create dataset adapters for DocVQA, ChartQA, TextVQA, and VQA v2 subsets.
- Document dataset licensing and expected directory layout.

## Phase 2: Shared Baseline

- Select a practical baseline VLM.
- Add inference-only baseline evaluation first.
- Add minimal training or fine-tuning hooks only after data contracts are stable.

## Phase 3: LoRA Fine-Tuning

- Add PEFT configuration.
- Track LoRA rank, target modules, memory usage, and task-level performance.
- Compare against the shared baseline.

## Phase 4: Task-Specific LoRA Experts

- Freeze one shared VLM backbone.
- Train separate LoRA experts for ChartQA, DocVQA, and TextVQA.
- Keep the backbone fixed so only adapter weights differ across tasks.
- Compare specialization gains and failure modes.

## Phase 5: Router-Based Adapter Selection

- Implement a task router that predicts `chartqa`, `docvqa`, or `textvqa`.
- Use the predicted task type to select `LoRA_chartqa`, `LoRA_docvqa`, or `LoRA_textvqa`.
- Do not route to separate full VLMs.
- Add instruction-aware routing features.
- Compare hard adapter routing, soft adapter weighting, and shared LoRA baselines.
- Report routing accuracy and answer accuracy together.

## Phase 6: Evaluation and Analysis

- Add per-task reports.
- Track latency and memory usage.
- Create experiment summaries for recruiter- and research-friendly presentation.

## Phase 7: Deployment

- Add FastAPI inference.
- Add Docker configuration.
- Add a simple Streamlit or React demo.
- Prepare model cards and usage notes.
