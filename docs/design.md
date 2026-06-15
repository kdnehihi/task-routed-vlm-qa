# System Design

## Overview

The intended system is a multi-task vision-language assistant that answers questions over images, documents, charts, and scientific or medical-style figures. The main architecture uses one shared frozen VLM backbone and multiple task-specific LoRA adapters. The router selects the LoRA adapter, not a completely different model.

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

### Shared Frozen VLM Backbone

A shared frozen backbone provides the common vision-language representation for all task types. The preferred main backbone is Qwen2.5-VL-7B-Instruct because its zero-shot behavior is strong on the current multi-task samples. Smaller baselines such as BLIP-VQA or GIT-VQA can still be used for comparison or low-resource experiments.

### Shared LoRA Baseline

A shared LoRA adapter tests whether one parameter-efficient adapter can improve multi-task performance without explicit routing.

### Task-Specific LoRA Experts

Task-specific LoRA adapters allow specialization while keeping the same frozen backbone:

- `LoRA_chartqa`
- `LoRA_docvqa`
- `LoRA_textvqa`

These experts are initially symbolic design targets. They should later be trained separately on their corresponding task subsets while the Qwen2-VL backbone remains frozen. This stage should compare specialization against the shared LoRA baseline without adding separate full models.

### Instruction/Task Router

The router predicts the task type from the question, instruction, and optionally visual metadata:

- `chartqa`
- `docvqa`
- `textvqa`

The predicted task type selects the corresponding LoRA adapter. The backbone remains the same.

```text
input image + question/instruction
-> instruction/task router
-> predicted task_type
-> selected LoRA adapter
-> same frozen VLM backbone + selected adapter
-> generated answer
```

The router should eventually be evaluated as a model component, not only as an implementation detail.

During router training and evaluation, each routing decision should be logged
with the selected expert id and adapter name. This makes it possible to inspect
whether the router is actually separating tasks or collapsing to one expert.

Example log format:

```text
[router] sample=42 expert=1 task=chartqa adapter=LoRA_chartqa true_task=chartqa correct=True confidence=0.913
```

The initial hard-routing design uses:

- expert `1`: `LoRA_chartqa`
- expert `2`: `LoRA_docvqa`
- expert `3`: `LoRA_textvqa`

Future experiments may add shared LoRA experts, such as an OCR-heavy shared
adapter for DocVQA/TextVQA or a general reasoning shared adapter for all tasks.
Those shared experts should be measured separately from task-specific experts
so their contribution is understandable.

## Planned Training Stages

1. Normalize datasets into a shared format.
2. Evaluate a pretrained shared VLM baseline.
3. Add a shared LoRA baseline for parameter-efficient adaptation.
4. Train task-specific LoRA experts on the same frozen backbone.
5. Add instruction-aware task routing for adapter selection.
6. Add optional MoE-style soft adapter weighting or expert composition.
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
