"""Routed Qwen2.5-VL inference service."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.models.baseline_vlm import DEFAULT_QWEN25VL_MODEL_NAME
from src.ops.model_manifest import ServingManifest
from src.routing.task_router import (
    BASE_BACKEND_NAME,
    CHARTQA_BACKEND_NAME,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_MULTIMODAL_ROUTER_DIR,
    TEXTVQA_BACKEND_NAME,
    MultimodalDebertaClipTaskRouter,
    RouterDecision,
    select_task_backend_for_image,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHART_ADAPTER_PATH = (
    PROJECT_ROOT
    / "checkpoints/chart_dora_r8_a16_B_lr2e-5/chart_dora_r8_a16_B_lr2e-5"
)
DEFAULT_TEXT_ADAPTER_PATH = PROJECT_ROOT / "checkpoints/textvqa_lora/textvqa_lora"

CHARTQA_PROMPT_TEMPLATE = """Read the chart carefully.
Use the chart title, axis labels, legend, colors, categories, and values to answer the question.
If the question asks yes/no, answer only Yes or No.
Otherwise return only the final value, label, or short phrase.
Do not explain.
Do not include extra text.

Question: {question}
Answer:"""

DOCVQA_PROMPT_TEMPLATE = """Read the document and answer the question.
Return only the exact answer span.
Do not explain.

Question: {question}
Answer:"""

TEXTVQA_PROMPT_TEMPLATE = """Read the visible text in the image and answer the question.
Return the complete answer, preserving important words and numbers.
Do not list all visible words.
Do not explain.

Question: {question}
Answer:"""


@dataclass(frozen=True)
class RoutedPrediction:
    """One routed VLM prediction."""

    answer: str
    decision: RouterDecision


class RoutedVLMService:
    """Load router, Qwen backbone, and task adapters once for API inference."""

    def __init__(
        self,
        model_name: str = DEFAULT_QWEN25VL_MODEL_NAME,
        router_dir: str | Path = PROJECT_ROOT / DEFAULT_MULTIMODAL_ROUTER_DIR,
        chart_adapter_path: str | Path = DEFAULT_CHART_ADAPTER_PATH,
        text_adapter_path: str | Path = DEFAULT_TEXT_ADAPTER_PATH,
        device: str | None = None,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 384 * 28 * 28,
        require_adapters: bool = True,
        local_files_only: bool = False,
        load_in_4bit: bool = False,
    ) -> None:
        self.model_name = model_name
        self.router_dir = Path(router_dir)
        self.chart_adapter_path = Path(chart_adapter_path)
        self.text_adapter_path = Path(text_adapter_path)
        self.device = device
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.require_adapters = require_adapters
        self.local_files_only = local_files_only
        self.load_in_4bit = load_in_4bit

        self.router: MultimodalDebertaClipTaskRouter | None = None
        self.processor: Any | None = None
        self.model: Any | None = None
        self.loaded_adapters: set[str] = set()
        self.manifest: ServingManifest | None = None

    @classmethod
    def from_manifest(cls, manifest: ServingManifest) -> "RoutedVLMService":
        """Build a service from an explicit serving artifact manifest."""
        service = cls(
            model_name=manifest.model_name,
            router_dir=manifest.router_dir,
            chart_adapter_path=manifest.chart_adapter_path,
            text_adapter_path=manifest.text_adapter_path,
            min_pixels=manifest.min_pixels,
            max_pixels=manifest.max_pixels,
            require_adapters=manifest.require_adapters,
            local_files_only=manifest.local_files_only,
            load_in_4bit=manifest.load_in_4bit,
        )
        service.manifest = manifest
        return service

    def load(self) -> "RoutedVLMService":
        """Load all long-lived inference objects."""
        self.router = self._load_router()
        self._load_qwen_model()
        return self

    def predict(
        self,
        image_path: str,
        question: str,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    ) -> RoutedPrediction:
        """Route one image-question pair and generate an answer."""
        if self.model is None or self.processor is None:
            raise RuntimeError("RoutedVLMService is not loaded. Call load() first.")

        clean_question = question.strip()
        decision = select_task_backend_for_image(
            question=clean_question,
            image_path=image_path,
            router=self.router,
            min_confidence=min_confidence,
        )
        answer = self._generate_answer(image_path, clean_question, decision)
        return RoutedPrediction(answer=answer, decision=decision)

    def _load_router(self) -> MultimodalDebertaClipTaskRouter | None:
        classifier_path = self.router_dir / "multimodal_logreg.joblib"
        if not classifier_path.exists():
            return None
        return MultimodalDebertaClipTaskRouter.load(
            self.router_dir,
            device=self.device,
            local_files_only=self.local_files_only,
        )

    def _load_qwen_model(self) -> None:
        import torch
        from transformers import AutoProcessor

        model_class = self._get_qwen_vl_model_class()
        self.device = self.device or self._select_device(torch)
        torch_dtype = self._torch_dtype_for_device(torch, self.device)

        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
            local_files_only=self.local_files_only,
        )
        load_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": "auto" if self.device == "cuda" else None,
            "local_files_only": self.local_files_only,
        }
        if self.load_in_4bit:
            if self.device != "cuda":
                raise ValueError("4-bit loading requires device='cuda'.")
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs.pop("torch_dtype")

        base_model = model_class.from_pretrained(self.model_name, **load_kwargs)
        if self.device != "cuda":
            base_model.to(self.device)

        self.model = base_model
        self._load_task_adapters()
        self.model.eval()

    def _load_task_adapters(self) -> None:
        self._load_one_adapter(self.chart_adapter_path, "chart_lora")
        self._load_one_adapter(self.text_adapter_path, "textvqa_lora")

    def _load_one_adapter(self, adapter_path: Path, adapter_name: str) -> None:
        if not adapter_path.exists():
            if self.require_adapters:
                raise FileNotFoundError(f"Missing adapter checkpoint: {adapter_path}")
            return

        from peft import PeftModel

        if self.loaded_adapters:
            self.model.load_adapter(
                str(adapter_path),
                adapter_name=adapter_name,
                is_trainable=False,
            )
        else:
            self.model = PeftModel.from_pretrained(
                self.model,
                str(adapter_path),
                adapter_name=adapter_name,
                is_trainable=False,
            )
        self.loaded_adapters.add(adapter_name)

    def _generate_answer(
        self,
        image_path: str,
        question: str,
        decision: RouterDecision,
    ) -> str:
        import torch
        from qwen_vl_utils import process_vision_info

        prompt_template = self._prompt_template_for_decision(decision)
        max_new_tokens = self._max_new_tokens_for_decision(decision)
        prompt = prompt_template.format(question=question)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(Path(image_path))},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        adapter_name = self._adapter_name_for_decision(decision)
        context = self._adapter_context(adapter_name)
        with context:
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    repetition_penalty=1.1,
                )

        generated_trimmed = [
            output[len(input_ids):]
            for input_ids, output in zip(inputs.input_ids, output_ids)
        ]
        answer = self.processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
        )[0]
        return self._clean_answer(answer)

    def _adapter_context(self, adapter_name: str | None):
        if adapter_name is not None:
            if adapter_name not in self.loaded_adapters:
                raise RuntimeError(f"Adapter is not loaded: {adapter_name}")
            self.model.set_adapter(adapter_name)
            return nullcontext()

        if hasattr(self.model, "disable_adapter"):
            return self.model.disable_adapter()
        return nullcontext()

    def _adapter_name_for_decision(self, decision: RouterDecision) -> str | None:
        if decision.backend_name == CHARTQA_BACKEND_NAME:
            return "chart_lora"
        if decision.backend_name == TEXTVQA_BACKEND_NAME:
            return "textvqa_lora"
        if decision.backend_name == BASE_BACKEND_NAME:
            return None
        raise ValueError(f"Unsupported backend: {decision.backend_name}")

    def _prompt_template_for_decision(self, decision: RouterDecision) -> str:
        if decision.backend_name == CHARTQA_BACKEND_NAME:
            return CHARTQA_PROMPT_TEMPLATE
        if decision.backend_name == TEXTVQA_BACKEND_NAME:
            return TEXTVQA_PROMPT_TEMPLATE
        return DOCVQA_PROMPT_TEMPLATE

    def _max_new_tokens_for_decision(self, decision: RouterDecision) -> int:
        if decision.backend_name == CHARTQA_BACKEND_NAME:
            return 8
        if decision.backend_name == TEXTVQA_BACKEND_NAME:
            return 12
        return 16

    def _get_qwen_vl_model_class(self):
        if "qwen2.5" in self.model_name.lower():
            from transformers import Qwen2_5_VLForConditionalGeneration

            return Qwen2_5_VLForConditionalGeneration

        from transformers import Qwen2VLForConditionalGeneration

        return Qwen2VLForConditionalGeneration

    @staticmethod
    def _clean_answer(answer: str) -> str:
        cleaned = " ".join(str(answer).strip().split())
        return cleaned.strip(" ,.;:")

    @staticmethod
    def _select_device(torch_module) -> str:
        if torch_module.cuda.is_available():
            return "cuda"
        if (
            hasattr(torch_module.backends, "mps")
            and torch_module.backends.mps.is_available()
        ):
            return "mps"
        return "cpu"

    @staticmethod
    def _torch_dtype_for_device(torch_module, device: str):
        if device in {"cuda", "mps"}:
            return torch_module.float16
        return torch_module.float32
