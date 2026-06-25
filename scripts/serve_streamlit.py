"""Simple Streamlit demo for routed Qwen2.5-VL QA inference."""

from __future__ import annotations

import gc
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.baseline_vlm import Qwen2VLVQABaselineVLM
from src.routing.task_router import (
    BASE_BACKEND_NAME,
    CHARTQA_BACKEND_NAME,
    DEFAULT_DEBERTA_ROUTER_DIR,
    DEFAULT_MULTIMODAL_ROUTER_DIR,
    TEXTVQA_BACKEND_NAME,
    DebertaEmbeddingLogRegTaskRouter,
    MultimodalDebertaClipTaskRouter,
    RouterDecision,
    select_task_backend_for_image,
)


MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
CHART_ADAPTER_PATH = "checkpoints/chart_dora_r8_a16_B_lr2e-5/chart_dora_r8_a16_B_lr2e-5"
TEXT_ADAPTER_PATH = "checkpoints/textvqa_lora/textvqa_lora"
DRIVE_CHECKPOINT_ROOT = Path(
    "/content/drive/MyDrive/multi-task-moe-vlm-assistant/checkpoints"
)

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

CHARTQA_PROMPT_TEMPLATE = """Read the chart carefully.
Use the chart title, axis labels, legend, colors, categories, and values to answer the question.
If the question asks yes/no, answer only Yes or No.
Otherwise return only the final value, label, or short phrase.
Do not explain.
Do not include extra text.

Question: {question}
Answer:"""


def candidate_router_paths() -> list[Path]:
    """Return router checkpoint locations commonly used locally and on Colab."""
    return [
        Path(DEFAULT_MULTIMODAL_ROUTER_DIR),
        PROJECT_ROOT / DEFAULT_MULTIMODAL_ROUTER_DIR,
        DRIVE_CHECKPOINT_ROOT / "router/multimodal_deberta_clip_router",
    ]


def resolve_router_path(requested_path: str | Path) -> Path:
    """Prefer the user path, otherwise auto-detect the trained router checkpoint."""
    requested = Path(requested_path).expanduser()
    if (requested / "multimodal_logreg.joblib").exists() or (
        requested / "embedding_logreg.joblib"
    ).exists():
        return requested

    for candidate in candidate_router_paths():
        if (candidate / "multimodal_logreg.joblib").exists() or (
            candidate / "embedding_logreg.joblib"
        ).exists():
            return candidate
    return requested


def load_router(
    router_path: Path,
) -> MultimodalDebertaClipTaskRouter | DebertaEmbeddingLogRegTaskRouter | None:
    """Load the trained multimodal router, with text-only fallback support."""
    multimodal_classifier = router_path / "multimodal_logreg.joblib"
    if multimodal_classifier.exists():
        return MultimodalDebertaClipTaskRouter.load(router_path)

    text_classifier = router_path / "embedding_logreg.joblib"
    if not text_classifier.exists():
        return None
    return DebertaEmbeddingLogRegTaskRouter.load(router_path)


def router_checkpoint_status(router_path: Path) -> dict[str, bool]:
    """Return the router files the demo can load from a checkpoint directory."""
    return {
        "router_dir": router_path.exists(),
        "multimodal_logreg.joblib": (router_path / "multimodal_logreg.joblib").exists(),
        "embedding_logreg.joblib": (router_path / "embedding_logreg.joblib").exists(),
    }


def model_config_for_decision(decision: RouterDecision) -> dict:
    """Return Qwen wrapper config for the selected backend."""
    if decision.backend_name == CHARTQA_BACKEND_NAME:
        return {
            "adapter_path": CHART_ADAPTER_PATH,
            "adapter_name": "chart_lora",
            "prompt_template": CHARTQA_PROMPT_TEMPLATE,
            "max_new_tokens": 8,
            "repetition_penalty": 1.1,
        }
    if decision.backend_name == TEXTVQA_BACKEND_NAME:
        return {
            "adapter_path": TEXT_ADAPTER_PATH,
            "adapter_name": "textvqa_lora",
            "prompt_template": TEXTVQA_PROMPT_TEMPLATE,
            "max_new_tokens": 12,
            "repetition_penalty": 1.1,
        }
    if decision.backend_name == BASE_BACKEND_NAME:
        return {
            "adapter_path": None,
            "adapter_name": "base",
            "prompt_template": DOCVQA_PROMPT_TEMPLATE,
            "max_new_tokens": 16,
            "repetition_penalty": 1.1,
        }
    raise ValueError(f"Unsupported backend: {decision.backend_name}")


def unload_current_model() -> None:
    """Drop the current session model before loading another backend."""
    import streamlit as st

    st.session_state.pop("active_model", None)
    st.session_state.pop("active_backend", None)
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def get_or_load_model(decision: RouterDecision) -> Qwen2VLVQABaselineVLM:
    """Keep only one Qwen backend loaded in the current Streamlit session."""
    import streamlit as st

    if st.session_state.get("active_backend") == decision.backend_name:
        return st.session_state["active_model"]

    unload_current_model()
    config = model_config_for_decision(decision)
    model = Qwen2VLVQABaselineVLM(
        model_name=MODEL_NAME,
        adapter_path=config["adapter_path"],
        adapter_name=config["adapter_name"],
        prompt_template=config["prompt_template"],
        max_new_tokens=config["max_new_tokens"],
        repetition_penalty=config["repetition_penalty"],
    )
    st.session_state["active_backend"] = decision.backend_name
    st.session_state["active_model"] = model
    return model


def save_uploaded_image(uploaded_file) -> str:
    """Persist one uploaded image to a temporary local file."""
    suffix = Path(uploaded_file.name).suffix or ".png"
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    handle.write(uploaded_file.getvalue())
    handle.flush()
    handle.close()
    return handle.name


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Routed VLM QA", layout="centered")
    st.title("Routed Qwen2.5-VL QA")

    router_path = st.sidebar.text_input(
        "Router checkpoint",
        value=str(DEFAULT_MULTIMODAL_ROUTER_DIR),
    )
    st.sidebar.caption(f"Text-only fallback path: `{DEFAULT_DEBERTA_ROUTER_DIR}`")
    router_path_obj = resolve_router_path(router_path)
    if router_path_obj != Path(router_path).expanduser():
        st.sidebar.success(f"Using detected router: `{router_path_obj}`")
    st.sidebar.json(router_checkpoint_status(router_path_obj))
    allow_rule_fallback = st.sidebar.checkbox(
        "Allow rule fallback if router checkpoint is missing",
        value=False,
        help="Fallback only reads the question text, so it can route chart images to TextVQA.",
    )
    min_confidence = st.sidebar.slider(
        "Router confidence fallback",
        min_value=0.0,
        max_value=1.0,
        value=0.65,
        step=0.05,
    )
    if st.sidebar.button("Unload current Qwen model"):
        unload_current_model()
        st.sidebar.success("Model unloaded.")

    uploaded_image = st.file_uploader(
        "Image",
        type=["png", "jpg", "jpeg", "webp"],
    )
    question = st.text_input("Question")

    if not uploaded_image:
        st.info("Upload an image to start.")
        return
    st.image(uploaded_image, caption="Input image", width="stretch")

    if not question.strip():
        st.info("Enter a question.")
        return

    if st.button("Answer", type="primary"):
        image_path = save_uploaded_image(uploaded_image)
        with st.spinner("Loading router..."):
            router = load_router(router_path_obj)
            if router is None:
                if not allow_rule_fallback:
                    st.error(
                        "Router checkpoint is missing, so routing by image is disabled. "
                        "Copy or select the trained multimodal router checkpoint before answering."
                    )
                    st.code(
                        "\n".join(
                            [
                                str(router_path_obj / "multimodal_logreg.joblib"),
                                str(router_path_obj / "embedding_logreg.joblib"),
                            ]
                        )
                    )
                    st.stop()
                st.warning(
                    "Router checkpoint missing. Using the question-only rule fallback, "
                    "which may choose the wrong backend for chart images."
                )
            decision = select_task_backend_for_image(
                question,
                image_path,
                router=router,
                min_confidence=min_confidence,
            )

        st.write(
            {
                "task_type": decision.task_type,
                "backend": decision.backend_name,
                "use_adapter": decision.use_adapter,
                "adapter": decision.adapter_name,
                "confidence": decision.confidence,
            }
        )

        with st.spinner(f"Running {decision.backend_name}..."):
            model = get_or_load_model(decision)
            answer = model.predict(image_path, question)

        st.subheader("Answer")
        st.write(answer)


if __name__ == "__main__":
    main()
