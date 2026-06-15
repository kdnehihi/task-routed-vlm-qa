# Notebooks

Use `train_qwen2vl_lora_baseline.ipynb` as the single Colab entry point for
Qwen2-VL adapter diagnostics.

Use `qwen2vl_lora_rank_sweep.ipynb` to run prioritized LoRA rank/config sweeps
and compare saved ChartQA/hybrid reports before committing to a larger run.

Set `ADAPTER_MODE` and `TRAIN_ADAPTER` in the config cell:

- `zero_shot`, `TRAIN_ADAPTER=False`: evaluate the base Qwen2-VL path.
- `shared_lora_all_tasks`, `TRAIN_ADAPTER=True`: train the backward-compatible
  shared adapter on ChartQA + DocVQA + TextVQA.
- `shared_doc_text_lora`, `TRAIN_ADAPTER=True`: train only DocVQA + TextVQA.
- `chart_lora_only`, `TRAIN_ADAPTER=True`: train only ChartQA with chart prompt,
  chart generation settings, and `chart_hybrid_accuracy`.
- `hybrid`, `TRAIN_ADAPTER=False`: evaluate ChartQA with `chart_lora` and
  DocVQA/TextVQA with `shared_doc_text_lora`.

Default data prep is capped at 1000 examples total: 333 DocVQA, 333 ChartQA,
and 334 TextVQA.

Saved adapter paths:

- `outputs/checkpoints/qwen2vl/qwen25vl_7b/shared_doc_text_lora`
- `outputs/checkpoints/qwen2vl/qwen25vl_7b/chart_lora`
- `outputs/checkpoints/qwen2vl/qwen25vl_7b/shared_lora_all_tasks`

No GRPO, DPO, MoME, router training, teacher distillation, or new backbone is
included in this notebook.
