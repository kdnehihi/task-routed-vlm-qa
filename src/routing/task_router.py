"""Question router for task-aware backend selection.

The current project router should select the best backend per task, not blindly
attach one LoRA adapter to every task.

Current backend policy:
- chartqa -> ChartQA DoRA adapter
- docvqa -> base Qwen2.5-VL zero-shot, no adapter
- textvqa -> TextVQA LoRA adapter
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.data.answers import canonicalize_task_type


TASK_TYPES = ("chartqa", "docvqa", "textvqa")
DEFAULT_ROUTER_PATH = Path("checkpoints/router/question_router.joblib")
DEFAULT_DEBERTA_ROUTER_DIR = Path("checkpoints/router/deberta_question_router")
DEFAULT_MIN_CONFIDENCE = 0.65

BASE_BACKEND_NAME = "base_zero_shot"
CHARTQA_BACKEND_NAME = "chart_dora_r8_a16_B_lr2e-5"
TEXTVQA_BACKEND_NAME = "textvqa_lora_only"

CHARTQA_ADAPTER_NAME = "chart_dora"
TEXTVQA_ADAPTER_NAME = "textvqa_lora"

CHARTQA_CHECKPOINT_DIR = (
    "checkpoints/chart_dora_r8_a16_B_lr2e-5/"
    "chart_dora_r8_a16_B_lr2e-5"
)
TEXTVQA_CHECKPOINT_DIR = (
    "/content/drive/MyDrive/multi-task-moe-vlm-assistant/checkpoints/"
    "qwen2vl/qwen25vl_7b/textvqa_lora"
)

CHART_KEYWORDS = (
    "chart",
    "graph",
    "axis",
    "bar",
    "line",
    "plot",
    "legend",
    "value",
    "values",
    "revenue",
    "percentage",
    "percent",
    "highest",
    "lowest",
)
DOC_KEYWORDS = (
    "document",
    "receipt",
    "invoice",
    "form",
    "date",
    "expiration",
    "total",
    "address",
    "signature",
    "phone",
    "company",
)


@dataclass(frozen=True)
class RouterDecision:
    """A task-router output that can be logged during inference."""

    task_type: str
    backend_name: str
    use_adapter: bool
    expert_id: int | None = None
    adapter_name: str | None = None
    checkpoint_dir: str | None = None
    confidence: float | None = None


class TfidfLogRegTaskRouter:
    """Trainable question-only task classifier.

    Typical workflow:
    1. Build ``questions`` and ``labels`` from processed JSONL metadata.
    2. Call ``router.fit(questions, labels)``.
    3. Evaluate on a held-out split.
    4. Save with ``router.save(...)``.
    5. Load at inference time with ``TfidfLogRegTaskRouter.load(...)``.
    """

    def __init__(self, pipeline: Any | None = None) -> None:
        self.pipeline = pipeline

    def fit(self, questions: list[str], labels: list[str]) -> "TfidfLogRegTaskRouter":
        """Train the TF-IDF + Logistic Regression router."""
        if len(questions) != len(labels):
            raise ValueError("questions and labels must have the same length")
        if not questions:
            raise ValueError("training data is empty")
        bad_labels = set(labels) - set(TASK_TYPES)
        if bad_labels:
            raise ValueError(f"Unknown labels: {bad_labels}")

        self.pipeline = build_tfidf_logreg_pipeline()
        self.pipeline.fit(questions, labels)
        return self

    def predict(self, question: str) -> str:
        """Predict one canonical task type from a question string."""
        self._ensure_fitted()
        return str(self.pipeline.predict([question])[0])

    def predict_with_confidence(self, question: str) -> tuple[str, float | None]:
        """Predict task and max probability when available."""
        self._ensure_fitted()
        task_type = self.predict(question)
        classifier = self.pipeline.named_steps["classifier"]
        if not hasattr(classifier, "predict_proba"):
            return task_type, None

        probabilities = self.pipeline.predict_proba([question])[0]
        confidence = float(max(probabilities))
        return task_type, confidence

    def save(self, path: str | Path = DEFAULT_ROUTER_PATH) -> None:
        """Save the trained sklearn pipeline."""
        self._ensure_fitted()
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_ROUTER_PATH) -> "TfidfLogRegTaskRouter":
        """Load a trained sklearn pipeline."""
        import joblib

        return cls(pipeline=joblib.load(path))

    def _ensure_fitted(self) -> None:
        if self.pipeline is None:
            raise RuntimeError("Router is not fitted. Call fit(...) or load(...) first.")


class DebertaEmbeddingLogRegTaskRouter:
    """Question router using frozen DeBERTa embeddings + Logistic Regression.

    This matches ``notebooks/router_deberta.ipynb``. The DeBERTa encoder is used
    only as a text feature extractor; the lightweight sklearn classifier makes
    the task prediction.
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-small",
        max_length: int = 96,
        classifier: Any | None = None,
        tokenizer: Any | None = None,
        encoder: Any | None = None,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.classifier = classifier
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.device = device

    def fit(
        self,
        questions: list[str],
        labels: list[str],
    ) -> "DebertaEmbeddingLogRegTaskRouter":
        """Fit the Logistic Regression classifier on DeBERTa embeddings."""
        if len(questions) != len(labels):
            raise ValueError("questions and labels must have the same length")
        if not questions:
            raise ValueError("training data is empty")
        bad_labels = set(labels) - set(TASK_TYPES)
        if bad_labels:
            raise ValueError(f"Unknown labels: {bad_labels}")

        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        embeddings = self.encode_questions(questions)
        self.classifier = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=2.0,
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        )
        self.classifier.fit(embeddings, labels)
        return self

    def predict(self, question: str) -> str:
        """Predict one canonical task type from a question string."""
        self._ensure_classifier()
        embedding = self.encode_questions([question])
        return str(self.classifier.predict(embedding)[0])

    def predict_with_confidence(self, question: str) -> tuple[str, float | None]:
        """Predict task and max Logistic Regression probability."""
        self._ensure_classifier()
        embedding = self.encode_questions([question])
        task_type = str(self.classifier.predict(embedding)[0])
        if not hasattr(self.classifier, "predict_proba"):
            return task_type, None
        probabilities = self.classifier.predict_proba(embedding)[0]
        return task_type, float(max(probabilities))

    def encode_questions(
        self,
        questions: list[str],
        batch_size: int = 32,
    ):
        """Encode question strings into mean-pooled DeBERTa vectors."""
        self._ensure_encoder()

        import numpy as np
        import torch

        vectors = []
        self.encoder.eval()
        with torch.no_grad():
            for start in range(0, len(questions), batch_size):
                batch_questions = questions[start : start + batch_size]
                encoded = self.tokenizer(
                    batch_questions,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {
                    key: value.to(self.device)
                    for key, value in encoded.items()
                }
                outputs = self.encoder(**encoded)
                pooled = self._mean_pool(
                    outputs.last_hidden_state,
                    encoded["attention_mask"],
                )
                vectors.append(pooled.detach().cpu().numpy())

        return np.concatenate(vectors, axis=0)

    def save(self, path: str | Path = DEFAULT_DEBERTA_ROUTER_DIR) -> None:
        """Save encoder, tokenizer, sklearn classifier, and router metadata."""
        self._ensure_encoder()
        self._ensure_classifier()
        import json
        import joblib

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(path / "tokenizer")
        self.encoder.save_pretrained(path / "encoder")
        joblib.dump(self.classifier, path / "embedding_logreg.joblib")
        (path / "router_config.json").write_text(
            json.dumps(
                {
                    "model_type": "deberta_embedding_logistic_regression_router",
                    "model_name": self.model_name,
                    "max_length": self.max_length,
                    "task_types": list(TASK_TYPES),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        path: str | Path = DEFAULT_DEBERTA_ROUTER_DIR,
        device: str | None = None,
    ) -> "DebertaEmbeddingLogRegTaskRouter":
        """Load a saved DeBERTa embedding router directory."""
        import json
        import joblib
        import torch
        from transformers import AutoModel, AutoTokenizer

        path = Path(path)
        config_path = path / "router_config.json"
        metadata_path = path / "router_metadata.json"
        config = {}
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
        elif metadata_path.exists():
            config = json.loads(metadata_path.read_text(encoding="utf-8"))

        model_name = config.get("model_name") or config.get("base_model")
        model_name = model_name or "microsoft/deberta-v3-small"
        max_length = int(config.get("max_length", 96))
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        tokenizer_path = path / "tokenizer"
        encoder_path = path / "encoder"
        classifier_path = path / "embedding_logreg.joblib"
        if not classifier_path.exists():
            raise FileNotFoundError(f"Missing router classifier: {classifier_path}")

        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path) if tokenizer_path.exists() else model_name
        )
        encoder = AutoModel.from_pretrained(
            str(encoder_path) if encoder_path.exists() else model_name
        ).to(device)
        classifier = joblib.load(classifier_path)

        return cls(
            model_name=model_name,
            max_length=max_length,
            classifier=classifier,
            tokenizer=tokenizer,
            encoder=encoder,
            device=device,
        )

    def _ensure_encoder(self) -> None:
        if self.tokenizer is not None and self.encoder is not None:
            return

        import torch
        from transformers import AutoModel, AutoTokenizer

        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.encoder = AutoModel.from_pretrained(self.model_name).to(self.device)

    def _ensure_classifier(self) -> None:
        if self.classifier is None:
            raise RuntimeError("Router is not fitted. Call fit(...) or load(...) first.")

    @staticmethod
    def _mean_pool(last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
        summed = (last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom


def build_tfidf_logreg_pipeline():
    """Return the sklearn TF-IDF + Logistic Regression pipeline."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, Pipeline

    return Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "word",
                            TfidfVectorizer(
                                lowercase=True,
                                analyzer="word",
                                ngram_range=(1, 3),
                                min_df=1,
                                sublinear_tf=True,
                            ),
                        ),
                        (
                            "char",
                            TfidfVectorizer(
                                lowercase=True,
                                analyzer="char_wb",
                                ngram_range=(3, 5),
                                min_df=1,
                                sublinear_tf=True,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=2.0,
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def get_backend_for_task(
    task_type: str,
    confidence: float | None = None,
) -> RouterDecision:
    """Map a task type to the current best backend decision."""
    normalized_task = canonicalize_task_type(task_type)

    if normalized_task == "chartqa":
        return RouterDecision(
            task_type="chartqa",
            backend_name=CHARTQA_BACKEND_NAME,
            use_adapter=True,
            expert_id=1,
            adapter_name=CHARTQA_ADAPTER_NAME,
            checkpoint_dir=CHARTQA_CHECKPOINT_DIR,
            confidence=confidence,
        )

    if normalized_task == "docvqa":
        return RouterDecision(
            task_type="docvqa",
            backend_name=BASE_BACKEND_NAME,
            use_adapter=False,
            expert_id=None,
            adapter_name=None,
            checkpoint_dir=None,
            confidence=confidence,
        )

    if normalized_task == "textvqa":
        return RouterDecision(
            task_type="textvqa",
            backend_name=TEXTVQA_BACKEND_NAME,
            use_adapter=True,
            expert_id=3,
            adapter_name=TEXTVQA_ADAPTER_NAME,
            checkpoint_dir=TEXTVQA_CHECKPOINT_DIR,
            confidence=confidence,
        )

    raise ValueError(f"Unsupported router task type: {task_type!r}")


def base_fallback_decision(
    task_type: str = "unknown",
    confidence: float | None = None,
) -> RouterDecision:
    """Return a safe no-adapter fallback for uncertain router predictions."""
    return RouterDecision(
        task_type=task_type,
        backend_name=BASE_BACKEND_NAME,
        use_adapter=False,
        expert_id=None,
        adapter_name=None,
        checkpoint_dir=None,
        confidence=confidence,
    )


def route_task_from_instruction(question: str) -> str:
    """Return a rough task type from a question string.

    This rule-based router predicts only the task type. Backend selection happens
    later in ``get_backend_for_task``.
    """
    normalized_question = question.lower()

    if any(keyword in normalized_question for keyword in CHART_KEYWORDS):
        return "chartqa"
    if any(keyword in normalized_question for keyword in DOC_KEYWORDS):
        return "docvqa"

    return "textvqa"


def select_task_backend(
    question: str,
    router: Any | None = None,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> RouterDecision:
    """Route a question to the current best backend.

    Behavior:
    - if ``router`` is None, use rule-based task prediction
    - if ``router`` exists, use ML task prediction + confidence
    - if confidence is too low, fall back to base zero-shot
    - DocVQA never attaches a DocVQA adapter in the final router path
    """
    if router is None:
        task_type = route_task_from_instruction(question)
        confidence = None
    else:
        task_type, confidence = router.predict_with_confidence(question)
        if confidence is not None and confidence < min_confidence:
            return base_fallback_decision("unknown", confidence)

    return get_backend_for_task(task_type, confidence=confidence)


def select_lora_expert(
    question: str,
    confidence: float | None = None,
    router: Any | None = None,
) -> RouterDecision:
    """Backward-compatible wrapper for older call sites.

    Prefer ``select_task_backend`` in new code. This wrapper now returns a
    backend decision, so DocVQA may return ``use_adapter=False``.
    """
    decision = select_task_backend(question, router=router)
    if confidence is None:
        return decision
    return RouterDecision(
        task_type=decision.task_type,
        backend_name=decision.backend_name,
        use_adapter=decision.use_adapter,
        expert_id=decision.expert_id,
        adapter_name=decision.adapter_name,
        checkpoint_dir=decision.checkpoint_dir,
        confidence=confidence,
    )


def format_router_decision(
    decision: RouterDecision,
    sample_id: int | str | None = None,
    true_task_type: str | None = None,
) -> str:
    """Format one router decision for readable inference logs."""
    parts = ["[router]"]

    if sample_id is not None:
        parts.append(f"sample={sample_id}")

    parts.extend(
        [
            f"task={decision.task_type}",
            f"backend={decision.backend_name}",
            f"use_adapter={decision.use_adapter}",
        ]
    )

    if decision.expert_id is not None:
        parts.append(f"expert={decision.expert_id}")
    if decision.adapter_name is not None:
        parts.append(f"adapter={decision.adapter_name}")

    if decision.confidence is not None:
        parts.append(f"confidence={decision.confidence:.3f}")

    if true_task_type is not None:
        is_correct = decision.task_type == true_task_type
        parts.append(f"true_task={true_task_type}")
        parts.append(f"correct={is_correct}")

    return " ".join(parts)


def summarize_router_decisions(decisions: list[RouterDecision]) -> dict[int, int]:
    """Count how often each adapter expert id was selected.

    Base fallback decisions have ``expert_id=None`` and are skipped.
    """
    counts: dict[int, int] = {}

    for decision in decisions:
        if decision.expert_id is None:
            continue
        counts[decision.expert_id] = counts.get(decision.expert_id, 0) + 1

    return counts


def summarize_router_backends(decisions: list[RouterDecision]) -> dict[str, int]:
    """Count how often each backend was selected."""
    counts: dict[str, int] = {}

    for decision in decisions:
        counts[decision.backend_name] = counts.get(decision.backend_name, 0) + 1

    return counts
