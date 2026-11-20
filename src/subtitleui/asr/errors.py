"""Custom exception hierarchy for the SubtitleUI ASR transcription module."""

from __future__ import annotations


class ASRError(Exception):
    """Base exception for all local ASR transcription errors."""


class InvalidTimeRangeError(ASRError):
    """Raised when a user-provided start/end time format or range is invalid."""


class AudioInputError(ASRError):
    """Raised when the input audio/video file is missing, unreadable, or has no audio."""


class AudioExtractionError(ASRError):
    """Raised when FFmpeg/FFprobe is unavailable or fails to extract/verify audio."""


class ModelLoadError(ASRError):
    """Raised when the requested Whisper model cannot be resolved or loaded."""


class TranscriptionError(ASRError):
    """Raised when an unexpected runtime error occurs during speech recognition."""


class TranscriptionCancelledError(ASRError):
    """Raised when transcription is cancelled by the user or a cancellation token."""
