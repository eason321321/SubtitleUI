"""Custom exception hierarchy for the SubtitleUI Phase 3 translation module."""

from __future__ import annotations


class TranslationError(Exception):
    """Base exception for all translation-related failures."""


class ConfigError(TranslationError):
    """Raised when the translation configuration file is invalid or unreadable."""


class TranscriptReadError(TranslationError):
    """Raised when a Phase 2 ASR transcript JSON file cannot be read or validated."""


class SelectionError(TranslationError):
    """Raised when segment indices or time-range selection arguments are invalid."""


class BackendError(TranslationError):
    """Base exception for translation backend execution failures."""


class BackendUnavailableError(BackendError):
    """Raised when a backend service, dependency, or model cannot be reached or loaded."""


class BackendTimeoutError(BackendError):
    """Raised when a translation backend request exceeds its configured timeout."""


class AlignmentError(TranslationError):
    """Raised when backend output fails strict segment index alignment validation."""


class TranslationCancelledError(TranslationError):
    """Raised when an ongoing translation workflow is cancelled by the user."""


class TranslationStorageError(TranslationError):
    """Raised when reading, merging, or writing translation JSON output fails."""
