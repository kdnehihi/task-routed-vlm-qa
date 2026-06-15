"""Train and evaluate a local ChartQA-only Qwen2-VL LoRA adapter.

This is intentionally separate from the Colab notebook so ChartQA LoRA design
can be tested locally on a small 400/100 split before spending notebook quota.
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import random
import re
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.answers import canonicalize_task_type, choose_training_answer
from src.evaluation.evaluator import build_prediction_records, summarize_quality_records_by_task


CHART_PROMPT = """Read the chart carefully.
Use the chart title, axis labels, legend, colors, categories, and values to answer the question.
If the question asks yes/no, answer only Yes or No.
Otherwise return only the final value, label, or short phrase.
Do not explain.
Do not include extra text.

Question: {question}
Answer:"""

TARGET_MODULE_PRESETS = {
    "A": ["q_proj", "v_proj"],
    "B": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "C": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--metadata-path", default="data/processed/multitask/validation.jsonl")
    parser.add_argument("--raw-chart-path", default="data/raw/chartqa/sample/validation.jsonl")
    parser.add_argument("--prepare-if-missing", action="store_true")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--sample-limit", type=int, default=500)
    parser.add_argument("--train-size", type=int, default=400)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--qlora", action="store_true", help="Load the base model in 4-bit NF4 for QLoRA.")
    parser.add_argument(
        "--bnb-4bit-compute-dtype",
        choices=["bf16", "fp16"],
        default="bf16",
        help="Compute dtype for 4-bit QLoRA layers.",
    )
    parser.add_argument("--target-modules-preset", choices=sorted(TARGET_MODULE_PRESETS), default="B")
    parser.add_argument("--target-modules", default=None)
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=512 * 28 * 28)
    parser.add_argument("--output-dir", default="outputs/checkpoints/qwen2vl/chart_lora_local")
    parser.add_argument("--predictions-path", default="outputs/predictions/qwen2vl_chart_lora_local_quality.jsonl")
    parser.add_argument("--report-path", default="outputs/predictions/qwen2vl_chart_lora_local_report.json")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--skip-label-debug", action="store_true")
    parser.add_argument("--print-limit", type=int, default=20)
    return parser.parse_args()


def ensure_dependencies(args) -> None:
    missing = []
    for module_name in ("peft", "qwen_vl_utils", "transformers", "torch"):
        try:
            __import__(module_name)
        except ModuleNotFoundError:
            missing.append(module_name)
    if args.qlora:
        try:
            __import__("bitsandbytes")
        except ModuleNotFoundError:
            missing.append("bitsandbytes")
    if missing:
        raise ModuleNotFoundError(
            "Missing required modules: "
            + ", ".join(missing)
            + ". Install project notebook dependencies before running this script."
        )


def prepare_chart_data(args) -> None:
    if not args.prepare_if_missing:
        return

    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/prepare_data.py"),
        "--dataset",
        "chartqa",
        "--split",
        "validation",
        "--limit",
        str(args.sample_limit),
    ]
    if args.streaming:
        command.append("--streaming")
    print("Preparing ChartQA data:", " ".join(command))
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def resolve_image_path(image_path: str) -> str:
    path = Path(image_path)
    if path.is_absolute():
        return str(path)
    return str(PROJECT_ROOT / path)


def load_chart_records(args) -> list[dict]:
    candidate_paths = [
        PROJECT_ROOT / args.raw_chart_path,
        PROJECT_ROOT / args.metadata_path,
    ]

    def read_chart_records_from(path: Path) -> list[dict]:
        records = []
        for record in read_jsonl(path):
            task_type = canonicalize_task_type(record.get("task_type"), record.get("dataset"))
            dataset = str(record.get("dataset", "")).lower()
            if task_type != "chartqa" and dataset != "chartqa":
                continue
            record["task_type"] = "chartqa"
            record["canonical_task_type"] = "chartqa"
            record["question"] = str(record.get("question") or "")
            record["answers"] = [str(answer) for answer in record.get("answers", [])]
            record["image_path"] = record["image_path"]
            record["chosen_training_answer"] = choose_training_answer(record["answers"], "chartqa")
            records.append(record)
        return records

    best_records = []
    best_source = None
    for source_path in candidate_paths:
        if not source_path.exists():
            continue
        records = read_chart_records_from(source_path)
        if len(records) > len(best_records):
            best_records = records
            best_source = source_path
        if len(records) >= args.sample_limit:
            return records[: args.sample_limit]

    if len(best_records) < args.sample_limit and args.prepare_if_missing:
        prepare_chart_data(args)
        raw_chart_path = PROJECT_ROOT / args.raw_chart_path
        if raw_chart_path.exists():
            records = read_chart_records_from(raw_chart_path)
            if len(records) >= args.sample_limit:
                return records[: args.sample_limit]
            best_records = records
            best_source = raw_chart_path

    if best_source is None:
        raise FileNotFoundError(
            "No metadata found. Run scripts/prepare_data.py for chartqa first, "
            "or pass --prepare-if-missing."
        )

    if len(best_records) < args.sample_limit:
        raise ValueError(
            f"Need {args.sample_limit} ChartQA records, found {len(best_records)} in {best_source}. "
            "Pass --prepare-if-missing, prepare more data, or lower --sample-limit."
        )
    return best_records[: args.sample_limit]


def split_records(records: list[dict], train_size: int, test_size: int, seed: int):
    if train_size + test_size > len(records):
        raise ValueError("train-size + test-size must be <= sample-limit.")
    shuffled = records.copy()
    random.Random(seed).shuffle(shuffled)
    train_records = shuffled[:train_size]
    test_records = shuffled[train_size:train_size + test_size]
    return train_records, test_records


def format_prompt(question: str) -> str:
    return CHART_PROMPT.format(question=question)


YES_NO_QUESTION_RE = re.compile(
    r"^\s*(is|are|was|were|do|does|did|can|could|has|have|had|will|would|"
    r"should|may|might)\b",
    re.IGNORECASE,
)


def is_yes_no_question(question: str) -> bool:
    return bool(YES_NO_QUESTION_RE.search(str(question)))


def clean_generated_answer(answer: str, question: str | None = None) -> str:
    cleaned = " ".join(str(answer).strip().split())
    if not cleaned:
        return cleaned
    if question is not None and not is_yes_no_question(question):
        cleaned = re.sub(r"\s+(?:yes|no)\s*$", "", cleaned, flags=re.IGNORECASE)
    tokens = cleaned.split()
    for start in range(1, len(tokens)):
        suffix = tokens[start:]
        if len(suffix) < 3:
            continue
        numeric_like = sum(
            bool(re.fullmatch(r"[\d.,:%$/-]+|[a-z]*\d+[a-z]*", token.lower()))
            for token in suffix
        )
        if numeric_like / len(suffix) >= 0.8:
            return " ".join(tokens[:start]).strip(" ,.;:")
    return cleaned.strip(" ,.;:")


def build_messages(record: dict, answer: str | None = None):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": resolve_image_path(record["image_path"])},
                {"type": "text", "text": format_prompt(record["question"])},
            ],
        }
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return messages


def find_subsequence(sequence: list[int], subsequence: list[int]) -> int:
    if not subsequence or len(subsequence) > len(sequence):
        return -1
    for start in range(len(sequence) - len(subsequence) + 1):
        if sequence[start:start + len(subsequence)] == subsequence:
            return start
    return -1


def mask_prompt_tokens(labels, full_inputs, prompt_inputs, row_index: int) -> None:
    full_ids = full_inputs["input_ids"][row_index].tolist()
    prompt_ids = prompt_inputs["input_ids"][row_index][prompt_inputs["attention_mask"][row_index].bool()].tolist()
    prompt_start = find_subsequence(full_ids, prompt_ids)
    if prompt_start < 0:
        raise ValueError(f"Could not align prompt tokens for row {row_index}; refusing unsafe labels.")
    labels[row_index, prompt_start:prompt_start + len(prompt_ids)] = -100


class ChartQADataset:
    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        answer = choose_training_answer(record.get("answers", []), "chartqa")
        return {
            "record": record,
            "answer": answer,
            "question": record["question"],
            "answers": record.get("answers", []),
        }


def build_collate_fn(processor, process_vision_info):
    def collate_fn(batch):
        full_messages = [build_messages(item["record"], item["answer"]) for item in batch]
        prompt_messages = [build_messages(item["record"]) for item in batch]
        answers = [item["answer"] for item in batch]
        questions = [item["question"] for item in batch]
        reference_answers = [item["answers"] for item in batch]

        full_texts = [
            processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            for messages in full_messages
        ]
        prompt_texts = [
            processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for messages in prompt_messages
        ]
        image_inputs, video_inputs = process_vision_info(full_messages)

        inputs = processor(
            text=full_texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        prompt_inputs = processor(
            text=prompt_texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        labels = inputs["input_ids"].clone()
        for row_index in range(labels.shape[0]):
            mask_prompt_tokens(labels, inputs, prompt_inputs, row_index)

        pad_token_id = processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100
        for token_id in set(processor.tokenizer.all_special_ids):
            labels[labels == token_id] = -100

        inputs["labels"] = labels
        inputs["questions"] = questions
        inputs["chosen_training_answers"] = answers
        inputs["reference_answers"] = reference_answers
        return inputs

    return collate_fn


def debug_decode_labels(processor, batch, n: int = 4) -> None:
    labels = batch["labels"].detach().cpu().clone()
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    pad_text = processor.tokenizer.pad_token or processor.tokenizer.eos_token or ""
    for index in range(min(n, labels.shape[0])):
        row = labels[index]
        row[row == -100] = pad_id
        decoded = processor.tokenizer.decode(row, skip_special_tokens=False)
        decoded_label = " ".join(decoded.replace(pad_text, " ").split())
        expected = batch["chosen_training_answers"][index]
        question = batch["questions"][index]
        print(f"[question {index}] {question!r}")
        print(f"[chosen answer {index}] {expected!r}")
        print(f"[decoded label {index}] {decoded_label!r}")
        assert decoded_label == expected
        assert question not in decoded_label
        for token in ("<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>", "<|image_pad|>"):
            assert token not in decoded_label


def model_device(model):
    return getattr(model, "device", next(model.parameters()).device)


def train(model, processor, process_vision_info, train_records: list[dict], args):
    import torch
    from torch.utils.data import DataLoader

    dataset = ChartQADataset(train_records)
    collate_fn = build_collate_fn(processor, process_vision_info)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    first_batch = next(iter(loader))
    if not args.skip_label_debug:
        debug_decode_labels(processor, first_batch)

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
    )
    use_cuda = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)
    autocast_device = "cuda" if use_cuda else "cpu"
    loss_history = []

    model.train()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        for step, batch in enumerate(loader, start=1):
            batch.pop("questions", None)
            batch.pop("chosen_training_answers", None)
            batch.pop("reference_answers", None)
            batch = {
                key: value.to(model_device(model)) if hasattr(value, "to") else value
                for key, value in batch.items()
            }
            with torch.autocast(device_type=autocast_device, dtype=torch.float16, enabled=use_cuda):
                outputs = model(**batch)
                loss = outputs.loss
            scaler.scale(loss / args.gradient_accumulation_steps).backward()
            if step % args.gradient_accumulation_steps == 0 or step == len(loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            loss_value = float(loss.detach().cpu())
            loss_history.append(loss_value)
            if step == 1 or step % 10 == 0:
                print(f"epoch={epoch + 1} step={step}/{len(loader)} loss={loss_value:.4f}")

    return {
        "overall_loss": sum(loss_history) / len(loss_history) if loss_history else None,
        "num_steps": len(loss_history),
    }


def predict_one(model, processor, process_vision_info, record: dict) -> tuple[str, str]:
    import torch

    messages = build_messages(record)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model_device(model))
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.1,
        )
    generated_trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    raw_prediction = processor.batch_decode(generated_trimmed, skip_special_tokens=True)[0].strip()
    return raw_prediction, clean_generated_answer(raw_prediction, record.get("question", ""))


def evaluate(model, processor, process_vision_info, test_records: list[dict], args):
    model.eval()
    raw_predictions = []
    cleaned_predictions = []
    for index, record in enumerate(test_records, start=1):
        raw_prediction, cleaned_prediction = predict_one(model, processor, process_vision_info, record)
        raw_predictions.append(raw_prediction)
        cleaned_predictions.append(cleaned_prediction)
        if index <= args.print_limit:
            print(f"[{index}/{len(test_records)}] {cleaned_prediction!r} answers={record['answers']}")

    quality_records = build_prediction_records(
        raw_predictions,
        test_records,
        cleaned_predictions=cleaned_predictions,
    )
    for record in quality_records:
        record["method"] = "chart_lora_local"
        record["adapter_backend"] = "chart_lora"
    return quality_records, summarize_quality_records_by_task(quality_records)


def target_modules_from_args(args) -> list[str]:
    if args.target_modules:
        return [module.strip() for module in args.target_modules.split(",") if module.strip()]
    return TARGET_MODULE_PRESETS[args.target_modules_preset]


def get_qwen_vl_model_class(model_name: str):
    """Return the right Transformers class for Qwen2-VL or Qwen2.5-VL."""
    if "qwen2.5" in model_name.lower():
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
        except ImportError as error:
            raise ImportError(
                "Qwen2.5-VL requires a recent transformers version with "
                "Qwen2_5_VLForConditionalGeneration. Upgrade transformers."
            ) from error
        return Qwen2_5_VLForConditionalGeneration

    from transformers import Qwen2VLForConditionalGeneration

    return Qwen2VLForConditionalGeneration


def main() -> None:
    args = parse_args()
    ensure_dependencies(args)

    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor
    from transformers import BitsAndBytesConfig

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    records = load_chart_records(args)
    train_records, test_records = split_records(records, args.train_size, args.test_size, args.seed)
    print(f"Loaded ChartQA records: total={len(records)} train={len(train_records)} test={len(test_records)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_dtype = torch.bfloat16 if args.bnb_4bit_compute_dtype == "bf16" else torch.float16
    dtype = compute_dtype if device == "cuda" else torch.float32
    quantization_config = None
    if args.qlora:
        if device != "cuda":
            raise RuntimeError("--qlora requires a CUDA GPU because bitsandbytes 4-bit layers run on GPU.")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    processor = AutoProcessor.from_pretrained(
        args.model_name,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    model_class = get_qwen_vl_model_class(args.model_name)
    model = model_class.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        quantization_config=quantization_config,
        device_map="auto" if device == "cuda" else None,
    )
    if device != "cuda":
        model.to(device)

    output_dir = PROJECT_ROOT / args.output_dir
    adapter_path = PROJECT_ROOT / args.adapter_path if args.adapter_path else output_dir
    loss_summary = None

    if args.eval_only:
        model = PeftModel.from_pretrained(model, adapter_path, adapter_name="chart_lora", is_trainable=False)
        print(f"Loaded chart_lora from {adapter_path}")
    else:
        model.config.use_cache = False
        if args.qlora:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
            print(
                "QLoRA enabled:",
                {
                    "load_in_4bit": True,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_use_double_quant": True,
                    "bnb_4bit_compute_dtype": args.bnb_4bit_compute_dtype,
                },
            )
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules_from_args(args),
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config, adapter_name="chart_lora")
        model.print_trainable_parameters()
        print(
            "LoRA config:",
            {
                "r": args.lora_r,
                "alpha": args.lora_alpha,
                "dropout": args.lora_dropout,
                "lr": args.learning_rate,
                "target_modules": target_modules_from_args(args),
            },
        )
        loss_summary = train(model, processor, process_vision_info, train_records, args)
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
        print(f"Saved chart_lora checkpoint to {output_dir}")

    quality_records, quality_report = evaluate(model, processor, process_vision_info, test_records, args)

    predictions_path = PROJECT_ROOT / args.predictions_path
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with predictions_path.open("w", encoding="utf-8") as f:
        for record in quality_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    report = {
        "chart_lora_local": quality_report,
        "loss_summary": loss_summary,
        "config": {
            "sample_limit": args.sample_limit,
            "train_size": args.train_size,
            "test_size": args.test_size,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "qlora": args.qlora,
            "bnb_4bit_compute_dtype": args.bnb_4bit_compute_dtype if args.qlora else None,
            "target_modules": target_modules_from_args(args),
        },
        "paths": {
            "checkpoint": str(output_dir),
            "predictions": str(predictions_path),
            "report": str(PROJECT_ROOT / args.report_path),
        },
    }
    report_path = PROJECT_ROOT / args.report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(quality_report, indent=2))
    print(f"Saved predictions to {predictions_path}")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
