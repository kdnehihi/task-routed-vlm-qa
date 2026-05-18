# System Design

## Overview

The intended system is a multi-task vision-language assistant that answers questions over images, documents, charts, and scientific or medical-style figures. The design emphasizes staged research: start with a shared baseline, then add LoRA, task-specific adapters, and eventually a router or MoE adapter layer.

This document describes the target architecture only. It does not imply that the modules are implemented yet.

## Inputs

Each training or inference example is expected to include:

- Image or document path
- User question
- Optional task label
- Optional answer candidates or ground-truth answer
- Optional metadata such as dataset name, split, OCR tokens, or chart type

TODO: Finalize a common schema in `src/data/dataset.py`.

## Planned Components

### Shared Baseline VLM

A shared baseline will provide a reference point for all task types. Candidate models include BLIP-VQA, Qwen2-VL small, LLaVA-small, or a lightweight frozen vision encoder with a small language decoder.

### LoRA Fine-Tuned Baseline

LoRA fine-tuning will test whether parameter-efficient adaptation improves multi-task performance without introducing explicit routing.

### Task-Specific Adapters

Task-specific adapters will allow separate specialization for image VQA, document QA, chart QA, and scientific figure QA. This stage should make it easier to compare specialization against a single shared model.

### Router and MoE Adapters

The router will eventually select or weight adapters based on question text, visual metadata, predicted task type, or instruction embeddings. The MoE version may route examples to task experts or combine expert outputs.

The router should eventually be evaluated as a model component, not only as an implementation detail.

## Planned Training Stages

1. Normalize datasets into a shared format.
2. Train or evaluate a shared multi-task baseline.
3. Add LoRA fine-tuning for parameter-efficient adaptation.
4. Add task-specific adapters.
5. Add instruction-aware task routing.
6. Add MoE-style adapter selection or expert composition.
7. Evaluate answer quality, routing quality, cost, latency, and memory usage.

## Evaluation Design

The evaluation layer should report both global and task-specific metrics:

- Exact Match
- ANLS for document-style QA
- Task-level accuracy
- Routing accuracy
- Latency
- Memory usage

TODO: Define metric contracts in `src/evaluation/metrics.py`.

## Serving Design

The planned serving layer will expose a FastAPI endpoint that accepts an image and question, then returns:

- Predicted answer
- Predicted task type
- Router confidence or expert weights if available
- Model metadata
- Latency information

No serving logic is implemented yet.

