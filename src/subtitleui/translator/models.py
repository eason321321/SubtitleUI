"""Data models for Phase 3 translation configuration, requests, progress, and results."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

from .errors import TranslationCancelledError

DEFAULT_PROVIDER = "ct2-nllb"
SUPPORTED_PROVIDERS: tuple[str, ...] = (
    "ct2-nllb",
    "local-llm",
)

SUPPORTED_LOCAL_LLM_BACKENDS: tuple[str, ...] = (
    "ollama",
    "llama-server",
    "lm-studio",
    "openai-compatible",
)

SUPPORTED_NLLB_DEVICES: tuple[str, ...] = (
    "auto",
    "cuda",
    "cpu",
)

DEFAULT_SOURCE_LANGUAGE = "ko"
DEFAULT_TARGET_LANGUAGE = "zh-TW"
DEFAULT_NLLB_MODEL_REPO = "JustFrederik/nllb-200-distilled-600M-ct2-int8"
DEFAULT_LOCAL_LLM_BACKEND = "ollama"
DEFAULT_LOCAL_LLM_BASE_URL = "http://localhost:11434/v1"
DEFAULT_LOCAL_LLM_MODEL = "qwen3:4b"
DEFAULT_LOCAL_LLM_TIMEOUT = 120
DEFAULT_BATCH_SIZE = 10
DEFAULT_CONTEXT_BEFORE = 2
DEFAULT_CONTEXT_AFTER = 2

# Mapping from standard ISO/BCP-47 tags to FLORES-200 / NLLB-200 language codes.
NLLB_LANGUAGE_CODES: dict[str, str] = {
    "ko": "kor_Hang",
    "ko-kr": "kor_Hang",
    "kor": "kor_Hang",
    "kor_hang": "kor_Hang",
    "zh-tw": "zho_Hant",
    "zh-hant": "zho_Hant",
    "zh_tw": "zho_Hant",
    "zh_hant": "zho_Hant",
    "zh-hk": "zho_Hant",
    "zho_hant": "zho_Hant",
    "zh-cn": "zho_Hans",
    "zh-hans": "zho_Hans",
    "zh_cn": "zho_Hans",
    "zh_hans": "zho_Hans",
    "zho_hans": "zho_Hans",
    "en": "eng_Latn",
    "en-us": "eng_Latn",
    "eng_latn": "eng_Latn",
    "ja": "jpn_Jpan",
    "ja-jp": "jpn_Jpan",
    "jpn_jpan": "jpn_Jpan",
}


def resolve_nllb_language_code(lang: str) -> str:
    """Resolve a language code (e.g. `ko`, `zh-TW`) to an NLLB FLORES-200 code."""
    cleaned = lang.strip()
    if not cleaned:
        raise ValueError("語言代碼不可為空白。")
    mapped = NLLB_LANGUAGE_CODES.get(cleaned.lower())
    if mapped is not None:
        return mapped
    # Allow passing explicit FLORES-200 codes such as `fra_Latn` directly.
    if "_" in cleaned and len(cleaned.split("_")) == 2:
        return cleaned
    raise ValueError(
        f"不支援的 NLLB 語言代碼「{lang}」，請使用 ko、zh-TW、zh-CN、en、ja 或標準 FLORES-200 代碼（如 kor_Hang、zho_Hant）。"
    )


class TranslationStage(str, Enum):
    """Lifecycle stages emitted during a translation workflow."""

    VALIDATING = "validating"
    LOADING_MODEL = "loading_model"
    TRANSLATING = "translating"
    SAVING = "saving"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERROR = "error"


class SelectionMode(str, Enum):
    """Selection modes for choosing transcript segments to translate."""

    FULL = "full"
    SEGMENTS = "segments"
    TIME_RANGE = "time_range"


class ExistingTranslationPolicy(str, Enum):
    """How to handle an existing translation JSON file at the target output path."""

    MERGE = "merge"
    OVERWRITE = "overwrite"
    SKIP_EXISTING = "skip_existing"


@runtime_checkable
class CancellationTokenLike(Protocol):
    """Duck-typed cancellation token protocol compatible with Phase 2 and future GUI."""

    @property
    def is_cancelled(self) -> bool:
        ...


class TranslationCancellationToken:
    """Thread-safe cooperative cancellation token for CLI and future GUI workflows."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        """Signal that the ongoing translation should be cancelled."""
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        """Return True if cancellation has been requested."""
        return self._event.is_set()

    def raise_if_cancelled(self, message: str = "翻譯已被使用者取消。") -> None:
        """Raise TranslationCancelledError if cancellation was requested."""
        if self._event.is_set():
            raise TranslationCancelledError(message)


def check_cancelled(
    cancel_token: CancellationTokenLike | None,
    message: str = "翻譯已被使用者取消。",
) -> None:
    """Check any compatible cancellation token and raise TranslationCancelledError."""
    if cancel_token is not None and cancel_token.is_cancelled:
        raise TranslationCancelledError(message)


@dataclass(frozen=True)
class LocalLLMConfig:
    """Configuration for connecting to an external OpenAI-compatible local LLM API."""

    backend: str = DEFAULT_LOCAL_LLM_BACKEND
    base_url: str = DEFAULT_LOCAL_LLM_BASE_URL
    model: str = DEFAULT_LOCAL_LLM_MODEL
    api_key: str = ""
    api_key_env: str = ""
    credential_key: str = ""
    timeout: float = DEFAULT_LOCAL_LLM_TIMEOUT
    max_retries: int = 2
    temperature: float = 0.2
    disable_thinking: bool = True


@dataclass(frozen=True)
class CT2NLLBConfig:
    """Configuration for the local CTranslate2 NLLB-200 translation backend."""

    model_path: Path = Path("./models/nllb")
    model_repo: str = DEFAULT_NLLB_MODEL_REPO
    device: str = "auto"
    compute_type: str = "int8"
    fallback_to_cpu: bool = True
    beam_size: int = 4
    max_decoding_length: int = 256
    cpu_threads: int = 0


@dataclass(frozen=True)
class TranslationConfig:
    """Top-level translation configuration loaded from `config/config.yaml`."""

    provider: str = DEFAULT_PROVIDER
    local_llm: LocalLLMConfig = field(default_factory=LocalLLMConfig)
    ct2_nllb: CT2NLLBConfig = field(default_factory=CT2NLLBConfig)
    source_language: str = DEFAULT_SOURCE_LANGUAGE
    target_language: str = DEFAULT_TARGET_LANGUAGE
    batch_size: int = DEFAULT_BATCH_SIZE
    context_before: int = DEFAULT_CONTEXT_BEFORE
    context_after: int = DEFAULT_CONTEXT_AFTER


@dataclass(frozen=True)
class SourceSegment:
    """An immutable source ASR segment with a stable 0-based index."""

    index: int
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class SourceTranscript:
    """Read-only representation of a Phase 2 ASR transcript JSON file."""

    source_path: Path
    source_media: str
    language: str
    segments: tuple[SourceSegment, ...]
    raw_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BackendBatchItem:
    """A single indexed text item sent to a translation backend."""

    index: int
    text: str


@dataclass(frozen=True)
class BackendBatchRequest:
    """A batch of target items with optional sliding-window context."""

    items: tuple[BackendBatchItem, ...]
    context_before: tuple[BackendBatchItem, ...] = ()
    context_after: tuple[BackendBatchItem, ...] = ()
    source_language: str = DEFAULT_SOURCE_LANGUAGE
    target_language: str = DEFAULT_TARGET_LANGUAGE


@dataclass(frozen=True)
class TranslatedSegment:
    """A single translated segment preserving original index, timestamps, and Korean text."""

    index: int
    start: float
    end: float
    source_text: str
    translated_text: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize segment to a JSON-compatible dictionary."""
        return {
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "source_text": self.source_text,
            "translated_text": self.translated_text,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TranslatedSegment:
        """Deserialize segment from a dictionary."""
        return cls(
            index=int(data["index"]),
            start=float(data["start"]),
            end=float(data["end"]),
            source_text=str(data["source_text"]),
            translated_text=str(data["translated_text"]),
        )


@dataclass(frozen=True)
class TranslationMetadata:
    """Metadata describing the source transcript, backend, model, and run details."""

    source_transcript: str
    backend: str
    translation_model: str
    source_language: str
    target_language: str
    selection_mode: str
    translated_at: str
    elapsed_seconds: float
    device: str | None = None
    total_source_segments: int | None = None
    translated_segments_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata to a JSON-compatible dictionary."""
        payload: dict[str, Any] = {
            "source_transcript": self.source_transcript,
            "backend": self.backend,
            "translation_model": self.translation_model,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "selection_mode": self.selection_mode,
            "translated_at": self.translated_at,
            "elapsed_seconds": self.elapsed_seconds,
        }
        if self.device is not None:
            payload["device"] = self.device
        if self.total_source_segments is not None:
            payload["total_source_segments"] = self.total_source_segments
        if self.translated_segments_count is not None:
            payload["translated_segments_count"] = self.translated_segments_count
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TranslationMetadata:
        """Deserialize metadata from a dictionary."""
        total_src = data.get("total_source_segments")
        translated_cnt = data.get("translated_segments_count")
        return cls(
            source_transcript=str(data["source_transcript"]),
            backend=str(data["backend"]),
            translation_model=str(data["translation_model"]),
            source_language=str(data["source_language"]),
            target_language=str(data["target_language"]),
            selection_mode=str(data["selection_mode"]),
            translated_at=str(data["translated_at"]),
            elapsed_seconds=float(data["elapsed_seconds"]),
            device=str(data["device"]) if data.get("device") is not None else None,
            total_source_segments=int(total_src) if total_src is not None else None,
            translated_segments_count=(
                int(translated_cnt) if translated_cnt is not None else None
            ),
        )


@dataclass(frozen=True)
class TranslationResult:
    """Complete output of a translation run."""

    metadata: TranslationMetadata
    segments: tuple[TranslatedSegment, ...]
    output_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize translation result to a JSON-compatible dictionary."""
        return {
            "metadata": self.metadata.to_dict(),
            "segments": [seg.to_dict() for seg in self.segments],
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        output_path: Path | None = None,
    ) -> TranslationResult:
        """Deserialize translation result from a dictionary."""
        metadata = TranslationMetadata.from_dict(data["metadata"])
        segments = tuple(
            TranslatedSegment.from_dict(item) for item in data.get("segments", [])
        )
        return cls(
            metadata=metadata,
            segments=segments,
            output_path=output_path,
        )


@dataclass(frozen=True)
class TranslationRequest:
    """Input specification for translating a Phase 2 ASR transcript file."""

    transcript_path: Path
    segments_spec: str | None = None
    segment_indices: tuple[int, ...] | None = None
    start_time: float | str | None = None
    end_time: float | str | None = None
    output_path: Path | None = None
    target_language: str | None = None
    existing_policy: ExistingTranslationPolicy = ExistingTranslationPolicy.MERGE


@dataclass(frozen=True)
class TranslationProgress:
    """Structured progress snapshot emitted throughout translation stages."""

    stage: TranslationStage
    backend: str
    model_name: str
    batch_index: int = 0
    total_batches: int = 0
    completed_segments: int = 0
    total_segments: int = 0
    elapsed_seconds: float = 0.0
    message: str = ""


ProgressCallback = Callable[[TranslationProgress], None]
