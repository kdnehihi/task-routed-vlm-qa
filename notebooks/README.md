# Notebooks

Use `train_qwen2vl_lora_baseline.ipynb` for the current Qwen2-VL sanity check.
It runs the full flow in one pass: data prep, zero-shot evaluation, label
decode checks, shared LoRA training, shared LoRA evaluation, and a final
zero-shot vs shared-LoRA comparison report.

The MoE/router notebooks were removed until the shared LoRA baseline is stable.
