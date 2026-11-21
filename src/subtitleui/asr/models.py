"""Data models for local ASR configuration, progress updates, and results."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from ..paths import MODELS_DIR
from .errors import TranscriptionCancelledError

DEFAULT_MODEL_NAME = "medium"
DEFAULT_LANGUAGE = "ko"
DEFAULT_TASK = "transcribe"
DEFAULT_MODELS_DIR = MODELS_DIR

SUPPORTED_MODELS: tuple[str, ...] = (
    "small",
    "medium",
    "large-v3",
    "large-v3-turbo",
)

SUPPORTED_DEVICES: tuple[str, ...] = (
    "auto",
    "cuda",
    "cpu",
)


class TranscriptionStage(str, Enum):
    """Lifecycle stages emitted during a transcription workflow."""

    VALIDATING = "validating"
    EXTRACTING_AUDIO = "extracting_audio"
    LOADING_MODEL = "loading_model"
    TRANSCRIBING = "transcribing"
    SAVING = "saving"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERROR = "error"


class CancellationToken:
    """Thread-safe cooperative cancellation token for CLI and future GUI workflows."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        """Signal that the ongoing operation should be cancelled."""
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        """Return True if cancellation has been requested."""
        return self._event.is_set()

    def raise_if_cancelled(self, message: str = "語音辨識已被使用者取消。") -> None:
        """Raise TranscriptionCancelledError if cancellation was requested."""
        if self._event.is_set():
            raise TranscriptionCancelledError(message)


@dataclass(frozen=True)
class TimeRange:
    """Validated media time interval in seconds relative to the original video."""

    start: float
    end: float

    @property
    def duration(self) -> float:
        """Duration of the selected interval in seconds."""
        return round(self.end - self.start, 6)


@dataclass(frozen=True)
class HardwareConfig:
    """Resolved runtime hardware configuration for CTranslate2 / Faster-Whisper."""

    device: str
    compute_type: str
    fallback_used: bool = False
    fallback_reason: str | None = None


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for loading and caching a local Faster-Whisper model."""

    model_name: str = DEFAULT_MODEL_NAME
    device: str = "auto"
    compute_type: str | None = None
    download_root: Path = DEFAULT_MODELS_DIR
    cpu_threads: int = 0


@dataclass(frozen=True)
class TranscriptionRequest:
    """Input specification for a single local ASR transcription job."""

    audio_path: Path
    language: str = DEFAULT_LANGUAGE
    task: str = DEFAULT_TASK
    start_time: float | str | None = None
    end_time: float | str | None = None
    output_path: Path | None = None
    beam_size: int = 5
    vad_filter: bool = False


@dataclass(frozen=True)
class TranscriptionProgress:
    """Structured status snapshot emitted throughout transcription stages."""

    stage: TranscriptionStage
    model_name: str
    device: str | None = None
    compute_type: str | None = None
    elapsed_seconds: float = 0.0
    processed_audio_seconds: float | None = None
    clip_duration_seconds: float | None = None
    segments_count: int = 0
    message: str = ""


@dataclass(frozen=True)
class TranscriptSegment:
    """A single transcribed speech segment aligned to the original media timeline."""

    start: float
    end: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize segment to a JSON-compatible dictionary."""
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TranscriptSegment:
        """Deserialize segment from a dictionary."""
        return cls(
            start=float(data["start"]),
            end=float(data["end"]),
            text=str(data["text"]),
        )


@dataclass(frozen=True)
class TranscriptionMetadata:
    """Metadata describing the source media, model, range, and runtime details."""

    source: str
    model: str
    language: str
    task: str
    transcribed_at: str
    range_start: float
    range_end: float
    source_duration: float
    device: str
    compute_type: str
    detected_language: str | None = None
    language_probability: float | None = None
    elapsed_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata to a JSON-compatible dictionary.

        Only includes model-reported language confidence when provided by the model.
        """
        payload: dict[str, Any] = {
            "source": self.source,
            "model": self.model,
            "language": self.language,
            "task": self.task,
            "transcribed_at": self.transcribed_at,
            "range_start": self.range_start,
            "range_end": self.range_end,
            "source_duration": self.source_duration,
            "device": self.device,
            "compute_type": self.compute_type,
        }
        if self.detected_language is not None:
            payload["detected_language"] = self.detected_language
        if self.language_probability is not None:
            payload["language_probability"] = self.language_probability
        if self.elapsed_seconds is not None:
            payload["elapsed_seconds"] = self.elapsed_seconds
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TranscriptionMetadata:
        """Deserialize metadata from a dictionary."""
        lang_prob = data.get("language_probability")
        elapsed = data.get("elapsed_seconds")
        return cls(
            source=str(data["source"]),
            model=str(data["model"]),
            language=str(data["language"]),
            task=str(data.get("task", DEFAULT_TASK)),
            transcribed_at=str(data["transcribed_at"]),
            range_start=float(data["range_start"]),
            range_end=float(data["range_end"]),
            source_duration=float(data["source_duration"]),
            device=str(data["device"]),
            compute_type=str(data["compute_type"]),
            detected_language=(
                str(data["detected_language"])
                if data.get("detected_language") is not None
                else None
            ),
            language_probability=float(lang_prob) if lang_prob is not None else None,
            elapsed_seconds=float(elapsed) if elapsed is not None else None,
        )


@dataclass(frozen=True)
class TranscriptionResult:
    """Complete output of a transcription run with original-timeline segments."""

    metadata: TranscriptionMetadata
    segments: tuple[TranscriptSegment, ...]
    output_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full transcription result to a JSON-compatible dictionary."""
        return {
            "metadata": self.metadata.to_dict(),
            "segments": [segment.to_dict() for segment in self.segments],
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        output_path: Path | None = None,
    ) -> TranscriptionResult:
        """Deserialize a transcription result from a dictionary."""
        metadata = TranscriptionMetadata.from_dict(data["metadata"])
        segments = tuple(
            TranscriptSegment.from_dict(item) for item in data.get("segments", [])
        )
        return cls(
            metadata=metadata,
            segments=segments,
            output_path=output_path,
        )


ProgressCallback = Callable[[TranscriptionProgress], None]
