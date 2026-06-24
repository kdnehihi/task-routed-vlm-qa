"""Baseline vision-language model interfaces and wrappers."""

from abc import ABC, abstractmethod
from pathlib import Path


DEFAULT_BLIP_MODEL_NAME = "Salesforce/blip-vqa-base"
DEFAULT_QWEN25VL_MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_CHARTQA_ADAPTER_PATH = (
    "checkpoints/chart_dora_r8_a16_B_lr2e-5/"
    "chart_dora_r8_a16_B_lr2e-5"
)

CHARTQA_PROMPT_TEMPLATE = """Read the chart carefully.
Use the chart title, axis labels, legend, colors, categories, and values to answer the question.
If the question asks yes/no, answer only Yes or No.
Otherwise return only the final value, label, or short phrase.
Do not explain.
Do not include extra text.

Question: {question}
Answer:"""


class BaselineVLM(ABC):
    """Interface for baseline vision-language QA models."""

    @abstractmethod
    def predict(self, image_path: str, question: str) -> str:
        """Return an answer for one image-question pair."""
        raise NotImplementedError


class DummyBaselineVLM(BaselineVLM):
    """Deterministic placeholder model for testing evaluation pipelines."""

    def __init__(self, default_answer: str = "") -> None:
        self.default_answer = default_answer

    def predict(self, image_path: str, question: str) -> str:
        """Return a fixed answer without reading the image or question."""
        return self.default_answer


class BlipVQABaselineVLM(BaselineVLM):
    """BLIP VQA baseline wrapper using Hugging Face Transformers.

    The model and processor are loaded lazily on the first prediction so that
    tests and lightweight CLI flows can import this module without downloading
    or initializing model weights.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_BLIP_MODEL_NAME,
        device: str | None = None,
        max_new_tokens: int = 20,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.processor = None
        self.model = None

    def predict(self, image_path: str, question: str) -> str:
        """Generate an answer for one image-question pair."""
        self._ensure_loaded()

        import torch
        from PIL import Image

        image = Image.open(Path(image_path)).convert("RGB")
        inputs = self.processor(image, question, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
            )
        answer = self.processor.decode(output_ids[0], skip_special_tokens=True)

        return answer.strip()

    def _ensure_loaded(self) -> None:
        """Load BLIP processor and model if they are not already loaded."""
        if self.processor is not None and self.model is not None:
            return

        import torch
        from transformers import BlipForQuestionAnswering, BlipProcessor

        self.device = self.device or self._select_device(torch)
        self.processor = BlipProcessor.from_pretrained(self.model_name)
        self.model = BlipForQuestionAnswering.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def _select_device(self, torch_module) -> str:
        """Select an available inference device."""
        if torch_module.cuda.is_available():
            return "cuda"
        if (
            hasattr(torch_module.backends, "mps")
            and torch_module.backends.mps.is_available()
        ):
            return "mps"
        return "cpu"


class BlipLoRAVQABaselineVLM(BlipVQABaselineVLM):
    """BLIP VQA wrapper that loads a PEFT LoRA adapter checkpoint."""

    def __init__(
        self,
        adapter_path: str,
        model_name: str = DEFAULT_BLIP_MODEL_NAME,
        device: str | None = None,
        max_new_tokens: int = 20,
    ) -> None:
        super().__init__(
            model_name=model_name,
            device=device,
            max_new_tokens=max_new_tokens,
        )
        self.adapter_path = adapter_path

    def _ensure_loaded(self) -> None:
        """Load BLIP base weights plus a LoRA adapter if needed."""
        if self.processor is not None and self.model is not None:
            return

        import torch
        from peft import PeftModel
        from transformers import BlipForQuestionAnswering, BlipProcessor

        self.device = self.device or self._select_device(torch)
        self.processor = BlipProcessor.from_pretrained(self.model_name)
        base_model = BlipForQuestionAnswering.from_pretrained(self.model_name)
        self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
        self.model.to(self.device)
        self.model.eval()


class Qwen2VLVQABaselineVLM(BaselineVLM):
    """Qwen2/Qwen2.5-VL VQA wrapper with optional PEFT adapter loading."""

    def __init__(
        self,
        model_name: str = DEFAULT_QWEN25VL_MODEL_NAME,
        device: str | None = None,
        adapter_path: str | None = None,
        adapter_name: str = "chart_lora",
        prompt_template: str = CHARTQA_PROMPT_TEMPLATE,
        max_new_tokens: int = 8,
        repetition_penalty: float = 1.1,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 384 * 28 * 28,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.adapter_path = adapter_path
        self.adapter_name = adapter_name
        self.prompt_template = prompt_template
        self.max_new_tokens = max_new_tokens
        self.repetition_penalty = repetition_penalty
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.processor = None
        self.model = None

    def predict(self, image_path: str, question: str) -> str:
        """Generate a short answer for one image-question pair."""
        self._ensure_loaded()

        import torch
        from qwen_vl_utils import process_vision_info

        messages = self._build_messages(image_path, question)
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

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
                repetition_penalty=self.repetition_penalty,
            )

        generated_trimmed = [
            output[len(input_ids):]
            for input_ids, output in zip(inputs.input_ids, output_ids)
        ]
        answer = self.processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
        )[0]
        return self._clean_generated_answer(answer, question)

    def _ensure_loaded(self) -> None:
        """Load Qwen2-VL base weights and optional PEFT adapter lazily."""
        if self.processor is not None and self.model is not None:
            return

        import torch
        from peft import PeftModel
        from transformers import AutoProcessor

        model_class = self._get_qwen_vl_model_class()
        self.device = self.device or self._select_device(torch)
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        base_model = model_class.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            device_map="auto" if self.device == "cuda" else None,
        )
        if self.device != "cuda":
            base_model.to(self.device)

        if self.adapter_path:
            self.model = PeftModel.from_pretrained(
                base_model,
                self.adapter_path,
                adapter_name=self.adapter_name,
                is_trainable=False,
            )
            self.model.set_adapter(self.adapter_name)
        else:
            self.model = base_model
        self.model.eval()

    def _get_qwen_vl_model_class(self):
        if "qwen2.5" in self.model_name.lower():
            from transformers import Qwen2_5_VLForConditionalGeneration

            return Qwen2_5_VLForConditionalGeneration

        from transformers import Qwen2VLForConditionalGeneration

        return Qwen2VLForConditionalGeneration

    def _build_messages(self, image_path: str, question: str) -> list[dict]:
        prompt = self.prompt_template.format(question=question)
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(Path(image_path))},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def _clean_generated_answer(self, answer: str, question: str) -> str:
        del question
        cleaned = " ".join(str(answer).strip().split())
        return cleaned.strip(" ,.;:")

    def _select_device(self, torch_module) -> str:
        if torch_module.cuda.is_available():
            return "cuda"
        if (
            hasattr(torch_module.backends, "mps")
            and torch_module.backends.mps.is_available()
        ):
            return "mps"
        return "cpu"


def create_baseline_model(
    model_name: str = "dummy",
    model_id: str | None = None,
    device: str | None = None,
    adapter_path: str | None = None,
) -> BaselineVLM:
    """Create a baseline model wrapper by name.

    TODO:
    - Add Qwen2-VL or LLaVA wrappers if hardware allows.
    - Add config-driven model loading.
    """
    if model_name == "dummy":
        return DummyBaselineVLM()
    if model_name == "blip":
        return BlipVQABaselineVLM(
            model_name=model_id or DEFAULT_BLIP_MODEL_NAME,
            device=device,
        )
    if model_name == "blip_lora":
        if adapter_path is None:
            raise ValueError("--adapter-path is required for blip_lora")
        return BlipLoRAVQABaselineVLM(
            adapter_path=adapter_path,
            model_name=model_id or DEFAULT_BLIP_MODEL_NAME,
            device=device,
        )
    if model_name == "qwen2vl":
        return Qwen2VLVQABaselineVLM(
            model_name=model_id or DEFAULT_QWEN25VL_MODEL_NAME,
            device=device,
        )
    if model_name == "qwen2vl_chart_lora":
        return Qwen2VLVQABaselineVLM(
            model_name=model_id or DEFAULT_QWEN25VL_MODEL_NAME,
            device=device,
            adapter_path=adapter_path or DEFAULT_CHARTQA_ADAPTER_PATH,
            adapter_name="chart_lora",
        )

    raise ValueError(f"Unsupported baseline model: {model_name}")
