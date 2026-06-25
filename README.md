# Routed Qwen2.5-VL QA

OCR-heavy visual question answering system for **ChartQA**, **DocVQA**, and **TextVQA**.

The project started as a MoE-LoRA fine-tuning experiment, but the strongest final shape is a safer routed system:

```text
image + question
-> multimodal task router
-> chartqa / docvqa / textvqa
-> selected Qwen2.5-VL backend
-> short answer
```

The current router policy is intentionally task-specific:

| Predicted task | Backend |
|---|---|
| ChartQA | `Qwen/Qwen2.5-VL-7B-Instruct` + ChartQA DoRA |
| DocVQA | `Qwen/Qwen2.5-VL-7B-Instruct` zero-shot |
| TextVQA | `Qwen/Qwen2.5-VL-7B-Instruct` + TextVQA LoRA |

## Demo

### ChartQA

<p>
  <img src="assets/readme/demo-chartqa1.png" width="49%" alt="ChartQA input demo">
  <img src="assets/readme/demo-chartqa2.png" width="49%" alt="ChartQA routed answer demo">
</p>

### DocVQA

<p>
  <img src="assets/readme/demo-docvqa1.png" width="49%" alt="DocVQA input demo">
  <img src="assets/readme/demo-docvqa2.png" width="49%" alt="DocVQA routed answer demo">
</p>

### TextVQA

<p>
  <img src="assets/readme/demo-textvqa1.png" width="49%" alt="TextVQA input demo">
  <img src="assets/readme/demo-textvqa2.png" width="49%" alt="TextVQA routed answer demo">
</p>

## Current Results

Validation snapshot from the Qwen2.5-VL-7B routed experiments:

| Method / backend | ChartQA hybrid | DocVQA ANLS | TextVQA score | Text EM | Notes |
|---|---:|---:|---:|---:|---|
| Zero-shot Qwen2.5-VL-7B | 0.8229 | **0.8677** | 0.7474 | 0.7812 | Strong base model; best DocVQA backend |
| ChartQA DoRA `r8_a16_B_lr2e-5` | **0.8333** | - | - | - | Best ChartQA backend |
| TextVQA LoRA `r4_a8` | - | - | **0.7708** | **0.8047** | Best TextVQA backend |

Multimodal router validation:

| Router | Accuracy | Macro F1 | Eval support |
|---|---:|---:|---:|
| DeBERTa text embeddings + CLIP image embeddings + Logistic Regression | **1.0000** | **1.0000** | 922 |

Router confusion snapshot:

| True / Pred | ChartQA | DocVQA | TextVQA |
|---|---:|---:|---:|
| ChartQA | 307 | 0 | 0 |
| DocVQA | 0 | 307 | 0 |
| TextVQA | 0 | 0 | 308 |

The project compares by each task's primary metric, not by one global exact-match number:

- ChartQA: `chart_hybrid_accuracy`
- DocVQA: `docvqa_anls`
- TextVQA: `textvqa_vqa_score`, with `TextEM` and `TextF1` reported for diagnosis

## What Changed During Development

### 1. Data and task names were made safe

Early experiments showed that fine-tuning could degrade a strong zero-shot Qwen2.5-VL baseline if the data and labels were not controlled carefully. The preprocessing pipeline now standardizes task names:

```text
document_qa -> docvqa
chart_qa    -> chartqa
image_vqa   -> textvqa
```

The training target selector avoids destructive normalization. `choose_training_answer(...)`:

- does not concatenate multiple references
- keeps the original answer text when possible
- picks the most frequent normalized reference group
- returns the shortest original answer from that group
- maps no-answer labels to `unanswerable` only when they are the majority

This keeps supervised targets closer to what Qwen2.5-VL should generate.

### 2. Evaluation normalization was separated from training labels

`normalize_answer(...)` is used for metrics, not for constructing training targets. This matters for OCR-heavy QA because lowercasing or over-normalizing labels can damage:

- document names
- title casing
- IDs and codes
- percent and decimal formatting
- short TextVQA spans

### 3. DocVQA improved through post-processing, not adapter training

DocVQA stayed on the zero-shot Qwen2.5-VL path because it was already very strong after fixing the data/evaluation layer. The current DocVQA metric path includes conservative cleanup:

- Unicode quote/dash normalization
- whitespace and punctuation cleanup
- safe handling of dollar values, initials, dates, and hyphenated spans
- DocVQA-specific exact match and ANLS normalization
- conservative short-span matching for metric debugging

This is why DocVQA is routed to `base_zero_shot` instead of a DocVQA LoRA adapter.

### 4. ChartQA needed a separate adapter and balanced sampling

ChartQA has inconsistent answer surfaces: percentages, decimals, labels, yes/no comparisons, and numeric chart values. The project added rule-based ChartQA metadata and stratified sampling:

- `lookup_value`
- `extreme`
- `difference`
- `average`
- `sum_total`
- `ratio`
- `yes_no_compare`
- `counting`
- `label_text`
- `time_year`
- `percent_decimal`

The best ChartQA adapter is currently:

```text
chart_dora_r8_a16_B_lr2e-5
rank=8
alpha=16
target modules=q_proj,k_proj,v_proj,o_proj
learning_rate=2e-5
```

### 5. TextVQA benefits from a small LoRA

TextVQA fine-tuning mainly helped output style: answers became shorter and cleaner, while still preserving important words. The best current TextVQA checkpoint is a non-DoRA LoRA:

```text
textvqa_lora
rank=4
alpha=8
dropout=0.05
target modules=q_proj,v_proj
```

The TextVQA metric can be strict because it uses multiple human references and soft VQA scoring. The project reports `TextEM`, `TextF1`, `ANLS`, and `textvqa_vqa_score` together to avoid misreading a single metric.

## Architecture

```text
                      +-------------------------------+
image + question ---> | multimodal task router         |
                      | DeBERTa text + CLIP image      |
                      | Logistic Regression classifier |
                      +---------------+---------------+
                                      |
             +------------------------+-------------------------+
             |                        |                         |
          chartqa                  docvqa                   textvqa
             |                        |                         |
   Qwen2.5-VL + Chart DoRA     Qwen2.5-VL base       Qwen2.5-VL + Text LoRA
             |                        |                         |
        short chart answer      exact document span       visible-text answer
```

The serving code keeps the design explicit:

- Router code: `src/routing/task_router.py`
- Qwen routed service: `src/serving/routed_vlm.py`
- FastAPI entrypoint: `src/serving/api.py`
- Streamlit Colab demo: `notebooks/colab_streamlit_demo.ipynb`

## Checkpoints

Runtime expects this local layout:

```text
checkpoints/
  router/
    multimodal_deberta_clip_router/
      multimodal_logreg.joblib
      text_tokenizer/
      text_encoder/
      image_processor/
      image_encoder/
  chart_dora_r8_a16_B_lr2e-5/
    chart_dora_r8_a16_B_lr2e-5/
      adapter_config.json
      adapter_model.safetensors
  textvqa_lora/
    textvqa_lora/
      adapter_config.json
      adapter_model.safetensors
```

For Colab, the demo notebook copies the same layout from:

```text
/content/drive/MyDrive/multi-task-moe-vlm-assistant/checkpoints/
```

## Run The Colab Demo

Use the notebook:

```text
notebooks/colab_streamlit_demo.ipynb
```

The notebook:

1. clones or pulls the repo
2. installs dependencies
3. removes stale Colab `torchao` if needed
4. mounts Google Drive
5. copies router and adapter checkpoints into local `checkpoints/`
6. starts the Streamlit app
7. opens a tunnel for browser access

## FastAPI Direction

The FastAPI path is in progress for a cleaner deployment shape:

```text
startup:
  load router once
  load Qwen2.5-VL once
  load ChartQA and TextVQA adapters once

request:
  route task
  switch adapter or disable adapter
  generate answer
```

This avoids repeatedly loading Qwen2.5-VL for each request.

## Repository Layout

```text
assets/readme/        README screenshots
data/                 raw and processed VQA data
notebooks/            Colab and experiment notebooks
scripts/              data prep, training, and demo entrypoints
src/data/             answer selection, preprocessing, ChartQA sampling
src/evaluation/       task-specific metrics and JSONL reports
src/models/           Qwen2.5-VL wrappers and LoRA utilities
src/routing/          task router and backend selection
src/serving/          FastAPI and routed inference service
tests/                data, metric, router, and adapter tests
```

## Useful Commands

Install:

```bash
pip install -r requirements.txt
```

Run tests:

```bash
pytest
```

Run Streamlit demo locally:

```bash
streamlit run scripts/serve_streamlit.py
```

Run FastAPI skeleton:

```bash
python scripts/serve_api.py
```

## Research Takeaway

The main lesson is that adapter fine-tuning was not automatically better than a strong Qwen2.5-VL zero-shot baseline. The useful gains came from making the pipeline safer:

- clean task names
- stable training targets
- task-specific prompts
- label masking checks
- task-specific metrics
- balanced ChartQA sampling
- conservative DocVQA post-processing
- a multimodal router that selects the right backend

The final system is therefore not "one LoRA for everything". It is a routed VLM QA system that uses the strongest backend per task.
