"""Abstract base interface for pluggable SubtitleUI translation backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import (
    BackendBatchItem,
    BackendBatchRequest,
    CancellationTokenLike,
    DEFAULT_SOURCE_LANGUAGE,
    DEFAULT_TARGET_LANGUAGE,
)


class TranslationBackend(ABC):
    """Unified interface implemented by all translation backends (CT2-NLLB, Local LLM, etc.)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short backend identifier (e.g. `ct2-nllb`, `local-llm`)."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Active model identifier reported in metadata and CLI status."""

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Return True if the backend is initialized and ready to translate."""

    @property
    def device_info(self) -> str | None:
        """Optional runtime device/compute description (e.g. `cuda/int8` or `cpu/int8`)."""
        return None

    @property
    def supports_context(self) -> bool:
        """Return True if this backend utilizes surrounding context windows."""
        return False

    @property
    def fallback_notice(self) -> str | None:
        """Optional warning/notice when hardware fallback occurred during load."""
        return None

    @abstractmethod
    def load(self) -> None:
        """Initialize resources or load the model into memory."""

    @abstractmethod
    def unload(self) -> None:
        """Release loaded models or resources from memory/VRAM."""

    @abstractmethod
    def translate_batch(
        self,
        request: BackendBatchRequest,
        *,
        cancel_token: CancellationTokenLike | None = None,
    ) -> dict[int, str]:
        """Translate a batch of indexed items and return a mapping `{index: translated_text}`."""

    def translate_text(
        self,
        text: str,
        *,
        source_language: str = DEFAULT_SOURCE_LANGUAGE,
        target_language: str = DEFAULT_TARGET_LANGUAGE,
        cancel_token: CancellationTokenLike | None = None,
    ) -> str:
        """Translate a single text string through the standard batch interface."""
        if not text or not text.strip():
            return ""
        if not self.is_loaded:
            self.load()
        request = BackendBatchRequest(
            items=(BackendBatchItem(index=0, text=text),),
            source_language=source_language,
            target_language=target_language,
        )
        mapping = self.translate_batch(request, cancel_token=cancel_token)
        return mapping.get(0, "")
