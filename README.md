# Multi-Task MoE Vision-Language Assistant

Research-oriented scaffold for a multi-task vision-language question answering system with a shared frozen VLM backbone, task-specific LoRA adapters, and instruction-aware adapter routing.

## Current Status

This repository is scaffold-only.

It currently contains project structure, documentation, configuration placeholders, Python module placeholders, and TODOs for future development. It does not implement the core model, training loop, router, MoE logic, real fine-tuning, dataset download, or inference service yet.

## Motivation

Vision-language QA tasks are often grouped under one broad "visual question answering" umbrella, but different tasks require different kinds of reasoning:

- Natural image VQA often depends on object recognition and visual commonsense.
- Document QA often depends on OCR, layout understanding, and key-value extraction.
- Chart QA often depends on visual data reading, axis interpretation, and numerical reasoning.
- Scientific or medical-style figure QA may require domain-specific visual interpretation and careful uncertainty tracking.

The central research question is whether a multi-task system can benefit from specialization while remaining modular and efficient. Instead of routing each task to a completely different model, the main architecture uses one shared frozen VLM backbone and routes each input to a task-specific LoRA adapter.

## Research Idea

The planned system will compare:

1. Shared multi-task baseline
2. LoRA fine-tuned baseline
3. Task-specific LoRA adapters on the same frozen backbone
4. Router-selected LoRA experts on the same frozen backbone

The main target design is:

```text
image + question/instruction
-> instruction/task router
-> predicted task_type: chartqa, docvqa, or textvqa
-> select LoRA_chartqa, LoRA_docvqa, or LoRA_textvqa
-> run the same frozen VLM backbone with the selected LoRA adapter
-> generate answer
```

Example routing intuition:

- "What is the value of the revenue in 2021?" should likely route to a ChartQA path.
- "What is the expiration date on this document?" should likely route to a Document QA path.

## Task Types

- Image VQA
- Document QA
- Chart QA
- Medical-style or scientific figure QA

## Planned Architecture

The long-term architecture may include:

- One shared frozen VLM backbone, such as Qwen2-VL-2B-Instruct if hardware allows
- Baseline VLM evaluation across all tasks
- Shared LoRA fine-tuning through PEFT
- Task-specific LoRA experts: `LoRA_chartqa`, `LoRA_docvqa`, `LoRA_textvqa`
- Adapter router that selects a LoRA expert, not a separate whole model
- Instruction-aware routing
- Evaluation module for answer quality and router quality
- FastAPI inference service
- Docker deployment
- Streamlit or simple frontend demo

Candidate model directions to evaluate later:

- BLIP-VQA baseline
- Qwen2-VL small model if GPU resources allow
- LLaVA-small if feasible
- Frozen vision encoder plus small language decoder as a lightweight fallback

## Planned Datasets

Possible datasets:

- DocVQA
- ChartQA
- TextVQA
- VQA v2 subset
- Medical-style or scientific figure QA datasets later

Dataset downloading and preprocessing are intentionally not implemented yet.

## Planned Pipeline

```text
data preprocessing
-> pretrained VLM baseline
-> shared LoRA baseline
-> task-specific LoRA experts
-> instruction/task router for adapter selection
-> evaluation
-> FastAPI inference service
-> Docker deployment
-> Streamlit or simple frontend demo
-> experiment analysis
```

## Planned Evaluation

Metrics to track:

- Exact Match
- ANLS
- Task-level accuracy
- Routing accuracy
- Latency
- Memory usage

Experiment tracking may include:

- Weights & Biases
- Hydra config
- Checkpoint tracking
- Experiment table

## Planned Deployment

The deployment path is planned as:

- Local inference script
- FastAPI endpoint
- Docker image
- Optional GPU inference endpoint
- Streamlit or simple React demo

## Repository Layout

```text
configs/      Planned experiment configs
data/         Raw and processed dataset placeholders
docs/         Design notes, roadmap, and experiment tracking
notebooks/    Exploration notebooks
outputs/      Local generated outputs
scripts/      Future CLI entry points
src/          Project package
tests/        Placeholder tests for future contracts
```

## Development Principle

This repository should grow in small, reviewable research stages. Each stage should document assumptions, configs, metrics, and known limitations so future contributors and reviewers can understand both the implementation and the reasoning behind it.

See:

- [docs/design.md](docs/design.md)
- [docs/roadmap.md](docs/roadmap.md)
- [docs/experiments.md](docs/experiments.md)
