"""Pluggable translation backends for SubtitleUI Phase 3."""

from .base import TranslationBackend
from .ct2_nllb import CT2NLLBBackend, check_ct2_cuda_availability
from .local_llm import (
    LocalLLMBackend,
    build_translation_messages,
    parse_and_validate_llm_response,
)

__all__ = [
    "CT2NLLBBackend",
    "LocalLLMBackend",
    "TranslationBackend",
    "build_translation_messages",
    "check_ct2_cuda_availability",
    "parse_and_validate_llm_response",
]
