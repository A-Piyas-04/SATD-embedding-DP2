"""SATD inference used by the Gradio demo.

Matches notebooks/satd-classifications-optimized V2.ipynb:
  text -> Qwen3-Embedding-0.6B (L2-normalized) -> identification XGBoost
       -> if SATD -> categorization XGBoost
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import torch

EMBED_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
MAX_SEQ_LENGTH = 256
DEFAULT_ARTIFACT = "issue"

# Trained heads store integer class ids (see Pipeline train notebooks):
#   identification: 0 = Not-SATD, 1 = SATD
#   categorization: LabelEncoder order over C/D, DOC, REQ, TES
IDENTIFICATION_LABELS = {0: "Not-SATD", 1: "SATD"}
CATEGORIZATION_LABELS = {0: "C/D", 1: "DOC", 2: "REQ", 3: "TES"}

# Cue phrases for live highlighting only (not used by the model).
KEYWORD_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("TODO", re.compile(r"\bTODO\b", re.IGNORECASE)),
    ("FIXME", re.compile(r"\bFIXME\b", re.IGNORECASE)),
    ("hack", re.compile(r"\bhack\b", re.IGNORECASE)),
    ("workaround", re.compile(r"\bwork\s*-?around\b", re.IGNORECASE)),
    ("temporary", re.compile(r"\btemporary\b", re.IGNORECASE)),
    ("technical debt", re.compile(r"\btechnical\s+debt\b", re.IGNORECASE)),
    ("design debt", re.compile(r"\bdesign\s+debt\b", re.IGNORECASE)),
    ("no tests", re.compile(r"\bno\s+tests?\b", re.IGNORECASE)),
    ("missing tests", re.compile(r"\bmissing\s+tests?\b", re.IGNORECASE)),
    ("unit tests", re.compile(r"\bunit\s+tests?\b", re.IGNORECASE)),
    ("flaky", re.compile(r"\bflaky\b", re.IGNORECASE)),
    ("test coverage", re.compile(r"\btest\s+coverage\b", re.IGNORECASE)),
    ("outdated docs", re.compile(r"\boutdated\s+docs?\b", re.IGNORECASE)),
    ("documentation", re.compile(r"\bdocumentation\b", re.IGNORECASE)),
    ("README", re.compile(r"\bREADME\b", re.IGNORECASE)),
    ("requirement", re.compile(r"\brequirements?\b", re.IGNORECASE)),
    ("never implemented", re.compile(r"\bnever\s+implemented\b", re.IGNORECASE)),
    ("unfinished", re.compile(r"\bunfinished\b", re.IGNORECASE)),
    ("shortcut", re.compile(r"\bshortcut\b", re.IGNORECASE)),
    ("refactor", re.compile(r"\brefactor\b", re.IGNORECASE)),
]

# Pipeline A issue heads (same filenames as V2)
MODEL_NAMES = {
    "issue": (
        "identification_issue_qwen3_xgboost.joblib",
        "categorization_issue_qwen3_xgboost.joblib",
    ),
    "code_comment": (
        "identification_code_comment_qwen3_xgboost.joblib",
        "categorization_code_comment_qwen3_xgboost.joblib",
    ),
    "commit": (
        "identification_commit_qwen3_xgboost.joblib",
        "categorization_commit_qwen3_xgboost.joblib",
    ),
    "pull_request": (
        "identification_pull_request_qwen3_xgboost.joblib",
        "categorization_pull_request_qwen3_xgboost.joblib",
    ),
}

MODES = ("Title only", "Description only", "Title + Description")


def default_model_dir() -> Path:
    env = os.environ.get("SATD_MODEL_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent / "models"


def find_keyword_cues(text: str) -> list[str]:
    """Return cue labels found in text (order preserved, unique)."""
    found: list[str] = []
    seen: set[str] = set()
    for label, pattern in KEYWORD_PATTERNS:
        if label in seen:
            continue
        if pattern.search(text or ""):
            found.append(label)
            seen.add(label)
    return found


def highlight_keyword_cues(text: str) -> str:
    """HTML with matched cue spans marked (for Gradio Markdown)."""
    if not (text or "").strip():
        return "_No text yet._"

    spans: list[tuple[int, int, str]] = []
    for label, pattern in KEYWORD_PATTERNS:
        for match in pattern.finditer(text):
            spans.append((match.start(), match.end(), label))

    if not spans:
        escaped = (
            text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )
        return f"<pre style='white-space:pre-wrap;font-family:inherit'>{escaped}</pre>"

    # Prefer longer matches when overlapping
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    chosen: list[tuple[int, int, str]] = []
    occupied_until = -1
    for start, end, label in spans:
        if start < occupied_until:
            continue
        chosen.append((start, end, label))
        occupied_until = end
    chosen.sort(key=lambda s: s[0])

    parts: list[str] = []
    cursor = 0
    for start, end, _label in chosen:
        parts.append(
            text[cursor:start]
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        chunk = (
            text[start:end]
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        parts.append(
            f"<mark style='background:#ffe566;padding:0 2px;border-radius:3px'>{chunk}</mark>"
        )
        cursor = end
    parts.append(
        text[cursor:].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )
    return (
        "<pre style='white-space:pre-wrap;font-family:inherit'>"
        + "".join(parts)
        + "</pre>"
    )


@dataclass
class PredictResult:
    text_used: str
    identification: str
    category: Optional[str]
    final_label: str
    embed_seconds: float
    classify_seconds: float
    total_seconds: float
    device: str
    mode: str = "Title + Description"
    identification_probs: dict[str, float] = field(default_factory=dict)
    category_probs: Optional[dict[str, float]] = None
    keyword_cues: list[str] = field(default_factory=list)


class SatdClassifier:
    """Load encoder + heads once; classify single texts quickly for demo."""

    def __init__(
        self,
        model_dir: Optional[Path] = None,
        artifact: str = DEFAULT_ARTIFACT,
        max_seq_length: int = MAX_SEQ_LENGTH,
        use_fp16: bool = True,
    ):
        self.model_dir = Path(model_dir) if model_dir else default_model_dir()
        self.artifact = artifact
        self.max_seq_length = max_seq_length
        self.use_fp16 = use_fp16 and torch.cuda.is_available()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedder = None
        self.identification_model = None
        self.categorization_model = None
        self._loaded = False

    def load(self) -> str:
        if self._loaded:
            return self.status()

        if self.artifact not in MODEL_NAMES:
            raise ValueError(f"Unknown artifact: {self.artifact}")

        ident_name, categ_name = MODEL_NAMES[self.artifact]
        ident_path = self.model_dir / ident_name
        categ_path = self.model_dir / categ_name

        missing = [p for p in (ident_path, categ_path) if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                "Missing model file(s):\n  "
                + "\n  ".join(str(p) for p in missing)
                + f"\n\nPlace Pipeline A .joblib files in:\n  {self.model_dir}\n"
                "Or set SATD_MODEL_DIR to the folder that contains them."
            )

        from sentence_transformers import SentenceTransformer

        model_kwargs = {"torch_dtype": torch.float16} if self.use_fp16 else {}
        self.embedder = SentenceTransformer(
            EMBED_MODEL_ID,
            device=self.device,
            model_kwargs=model_kwargs if self.device == "cuda" else {},
        )
        self.embedder.max_seq_length = self.max_seq_length
        self.embedder.eval()

        self.identification_model = joblib.load(ident_path)
        self.categorization_model = joblib.load(categ_path)
        self._loaded = True
        return self.status()

    def status(self) -> str:
        if not self._loaded:
            return "Models not loaded yet."
        return (
            f"Ready | device={self.device} | artifact={self.artifact} | "
            f"fp16={self.use_fp16} | max_seq_length={self.max_seq_length} | "
            f"id_classes={[IDENTIFICATION_LABELS.get(int(c), c) for c in self.identification_model.classes_]} | "
            f"cat_classes={[CATEGORIZATION_LABELS.get(int(c), c) for c in self.categorization_model.classes_]}"
        )

    def build_text(self, title: str, description: str, mode: str) -> str:
        title = (title or "").strip()
        description = (description or "").strip()
        mode = (mode or "Title + Description").strip()

        if mode == "Title only":
            return title
        if mode == "Description only":
            return description
        return f"{title} {description}".strip()

    @staticmethod
    def _raw_label(value) -> object:
        if hasattr(value, "item"):
            return value.item()
        return value

    def _decode_identification(self, raw) -> str:
        value = self._raw_label(raw)
        if isinstance(value, str):
            key = value.strip()
            if key in {"SATD", "Not-SATD"}:
                return key
            if key in {"0", "1"}:
                return IDENTIFICATION_LABELS[int(key)]
            return key
        try:
            return IDENTIFICATION_LABELS[int(value)]
        except (KeyError, TypeError, ValueError):
            return str(value)

    def _decode_category(self, raw) -> str:
        value = self._raw_label(raw)
        if isinstance(value, str):
            key = value.strip()
            if key in CATEGORIZATION_LABELS.values():
                return key
            if key.isdigit():
                return CATEGORIZATION_LABELS.get(int(key), key)
            return key
        try:
            return CATEGORIZATION_LABELS[int(value)]
        except (KeyError, TypeError, ValueError):
            return str(value)

    def _identification_probs(self, emb: np.ndarray) -> dict[str, float]:
        proba = self.identification_model.predict_proba(emb)[0]
        return {
            IDENTIFICATION_LABELS.get(int(cls), str(cls)): float(p)
            for cls, p in zip(self.identification_model.classes_, proba)
        }

    def _category_probs(self, emb: np.ndarray) -> dict[str, float]:
        proba = self.categorization_model.predict_proba(emb)[0]
        return {
            CATEGORIZATION_LABELS.get(int(cls), str(cls)): float(p)
            for cls, p in zip(self.categorization_model.classes_, proba)
        }

    def classify_embedding(
        self, embedding: np.ndarray
    ) -> tuple[str, Optional[str], str, dict[str, float], Optional[dict[str, float]]]:
        """Two-step classify with probabilities."""
        emb = np.asarray(embedding, dtype=np.float32)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)

        id_probs = self._identification_probs(emb)
        identification = self._decode_identification(
            self.identification_model.predict(emb)[0]
        )
        if identification == "SATD":
            cat_probs = self._category_probs(emb)
            category = self._decode_category(
                self.categorization_model.predict(emb)[0]
            )
            return identification, category, category, id_probs, cat_probs
        return identification, None, identification, id_probs, None

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        with torch.inference_mode():
            return self.embedder.encode(
                texts,
                batch_size=len(texts),
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

    def predict(self, title: str, description: str, mode: str) -> PredictResult:
        if not self._loaded:
            self.load()

        text = self.build_text(title, description, mode)
        if not text:
            raise ValueError("Enter a Title and/or Description before classifying.")

        t0 = time.perf_counter()
        emb = self._embed_texts([text])
        t1 = time.perf_counter()

        identification, category, final_label, id_probs, cat_probs = (
            self.classify_embedding(emb)
        )
        t2 = time.perf_counter()

        return PredictResult(
            text_used=text,
            identification=identification,
            category=category,
            final_label=final_label,
            embed_seconds=t1 - t0,
            classify_seconds=t2 - t1,
            total_seconds=t2 - t0,
            device=self.device,
            mode=mode,
            identification_probs=id_probs,
            category_probs=cat_probs,
            keyword_cues=find_keyword_cues(text),
        )

    def predict_three_modes(
        self, title: str, description: str
    ) -> tuple[list[PredictResult], float]:
        """Classify Title / Description / Both in one embedding batch."""
        if not self._loaded:
            self.load()

        mode_texts: list[tuple[str, str]] = []
        for mode in MODES:
            text = self.build_text(title, description, mode)
            if text:
                mode_texts.append((mode, text))

        if not mode_texts:
            raise ValueError("Enter a Title and/or Description before classifying.")

        texts = [t for _, t in mode_texts]
        t0 = time.perf_counter()
        emb = np.asarray(self._embed_texts(texts), dtype=np.float32)
        t1 = time.perf_counter()

        results: list[PredictResult] = []
        for i, (mode, text) in enumerate(mode_texts):
            identification, category, final_label, id_probs, cat_probs = (
                self.classify_embedding(emb[i])
            )
            results.append(
                PredictResult(
                    text_used=text,
                    identification=identification,
                    category=category,
                    final_label=final_label,
                    embed_seconds=0.0,
                    classify_seconds=0.0,
                    total_seconds=0.0,
                    device=self.device,
                    mode=mode,
                    identification_probs=id_probs,
                    category_probs=cat_probs,
                    keyword_cues=find_keyword_cues(text),
                )
            )
        t2 = time.perf_counter()
        # Attach shared timing to each row for display convenience
        total = t2 - t0
        embed_s = t1 - t0
        classify_s = t2 - t1
        for r in results:
            r.embed_seconds = embed_s
            r.classify_seconds = classify_s
            r.total_seconds = total
        return results, total
