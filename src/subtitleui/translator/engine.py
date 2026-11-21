"""Core backend-agnostic TranslationEngine with batching, context windows, alignment recovery, and lifecycle management."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from ..paths import default_translation_output_path
from .backends.base import TranslationBackend
from .backends.ct2_nllb import CT2NLLBBackend
from .backends.local_llm import LocalLLMBackend
from .config import ConfigManager
from .errors import (
    AlignmentError,
    ConfigError,
    TranslationCancelledError,
    TranslationError,
)
from .models import (
    BackendBatchItem,
    BackendBatchRequest,
    CancellationTokenLike,
    ExistingTranslationPolicy,
    ProgressCallback,
    SelectionMode,
    SourceSegment,
    TranslatedSegment,
    TranslationConfig,
    TranslationMetadata,
    TranslationProgress,
    TranslationRequest,
    TranslationResult,
    TranslationStage,
    check_cancelled,
)
from .selector import select_segments
from .storage import (
    load_asr_transcript,
    load_translation_json,
    save_translation_json,
)


def create_backend(config: TranslationConfig) -> TranslationBackend:
    """Instantiate the configured TranslationBackend from TranslationConfig."""
    provider = config.provider.strip().lower()
    if provider == "ct2-nllb":
        return CT2NLLBBackend(config.ct2_nllb)
    if provider == "local-llm":
        return LocalLLMBackend(config.local_llm)
    raise ConfigError(
        f"不支援的翻譯後端「{config.provider}」，請使用 ct2-nllb 或 local-llm。"
    )


def _split_into_contiguous_batches(
    selected: Sequence[SourceSegment],
    batch_size: int,
) -> list[list[SourceSegment]]:
    """Group selected segments into contiguous runs, then chunk by `batch_size`.

    Keeping batches contiguous ensures every batch receives accurate preceding
    and following sliding-window context from the full transcript.
    """
    if not selected:
        return []
    size = max(1, int(batch_size))
    runs: list[list[SourceSegment]] = []
    current_run: list[SourceSegment] = []

    for seg in selected:
        if not current_run or seg.index == current_run[-1].index + 1:
            current_run.append(seg)
        else:
            runs.append(current_run)
            current_run = [seg]
    if current_run:
        runs.append(current_run)

    batches: list[list[SourceSegment]] = []
    for run in runs:
        for offset in range(0, len(run), size):
            batches.append(run[offset : offset + size])
    return batches


def _validate_batch_mapping(
    batch: Sequence[SourceSegment],
    mapping: dict[int, str],
) -> None:
    """Enforce strict 1-to-1 alignment verification in the engine regardless of backend."""
    if not isinstance(mapping, dict):
        raise AlignmentError(
            f"翻譯後端回傳型別錯誤（預期 dict，收到 {type(mapping).__name__}）。"
        )

    expected_ids = {seg.index for seg in batch}
    returned_ids = set(mapping.keys())

    unknown = sorted(returned_ids - expected_ids)
    if unknown:
        raise AlignmentError(f"翻譯結果包含不存在的句子索引：{unknown}")

    missing = sorted(expected_ids - returned_ids)
    if missing:
        raise AlignmentError(f"翻譯結果漏翻句子索引：{missing}")

    for seg in batch:
        val = mapping[seg.index]
        if not isinstance(val, str):
            raise AlignmentError(
                f"句子索引 {seg.index} 的翻譯結果必須是字串，目前為 {type(val).__name__}。"
            )
        if seg.text.strip() and not val.strip():
            raise AlignmentError(
                f"句子索引 {seg.index} 的原文非空白，但翻譯結果為空白。"
            )


class TranslationEngine:
    """High-level translation orchestrator decoupled from specific models and ASR internals."""

    def __init__(
        self,
        config: TranslationConfig | None = None,
        *,
        backend: TranslationBackend | None = None,
        config_manager: ConfigManager | None = None,
        single_item_retries: int = 2,
        before_load_hook: Callable[[], None] | None = None,
    ) -> None:
        if config is not None:
            self._config = config
        elif config_manager is not None:
            self._config = config_manager.load(auto_create=True)
        else:
            self._config = TranslationConfig()

        self._backend: TranslationBackend = backend or create_backend(self._config)
        self._single_item_retries = max(0, int(single_item_retries))
        self._before_load_hook = before_load_hook
        self._current_stage: TranslationStage | None = None

    @property
    def config(self) -> TranslationConfig:
        """Return active translation configuration."""
        return self._config

    @property
    def backend(self) -> TranslationBackend:
        """Return active translation backend."""
        return self._backend

    @property
    def is_loaded(self) -> bool:
        """Return True if the active backend is loaded and ready."""
        return self._backend.is_loaded

    @property
    def current_stage(self) -> TranslationStage | None:
        """Return the most recently entered workflow stage."""
        return self._current_stage

    def load_backend(
        self,
        *,
        progress_callback: ProgressCallback | None = None,
        cancel_token: CancellationTokenLike | None = None,
        _started_at: float | None = None,
    ) -> None:
        """Load the active translation backend once, invoking any VRAM release hook first."""
        if self._backend.is_loaded:
            return

        check_cancelled(cancel_token)
        started_at = _started_at if _started_at is not None else time.monotonic()

        if self._before_load_hook is not None:
            self._before_load_hook()

        self._emit_progress(
            progress_callback,
            TranslationProgress(
                stage=TranslationStage.LOADING_MODEL,
                backend=self._backend.name,
                model_name=self._backend.model_name,
                elapsed_seconds=round(time.monotonic() - started_at, 2),
                message=(
                    f"正在初始化翻譯後端「{self._backend.name}」"
                    f"（模型：{self._backend.model_name}）..."
                ),
            ),
        )
        self._backend.load()

        if self._backend.fallback_notice:
            self._emit_progress(
                progress_callback,
                TranslationProgress(
                    stage=TranslationStage.LOADING_MODEL,
                    backend=self._backend.name,
                    model_name=self._backend.model_name,
                    elapsed_seconds=round(time.monotonic() - started_at, 2),
                    message=self._backend.fallback_notice,
                ),
            )

    def unload_backend(self) -> None:
        """Release the active backend model from VRAM/RAM."""
        self._backend.unload()

    def set_backend(self, backend: TranslationBackend) -> None:
        """Switch to another TranslationBackend, unloading the current one first."""
        if self._backend is not backend:
            self._backend.unload()
            self._backend = backend

    def __enter__(self) -> TranslationEngine:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.unload_backend()

    def translate_text(
        self,
        text: str,
        *,
        source_language: str | None = None,
        target_language: str | None = None,
        cancel_token: CancellationTokenLike | None = None,
    ) -> str:
        """Pure in-memory API for translating a single string (for GUI / Phase 7)."""
        check_cancelled(cancel_token)
        if not text or not text.strip():
            return ""
        self.load_backend(cancel_token=cancel_token)
        src_lang = source_language or self._config.source_language
        tgt_lang = target_language or self._config.target_language
        seg = SourceSegment(index=0, start=0.0, end=0.0, text=text)
        mapping = self._translate_batch_with_recovery(
            batch=[seg],
            all_segments_by_index={0: seg},
            total_source_count=1,
            source_language=src_lang,
            target_language=tgt_lang,
            cancel_token=cancel_token,
        )
        return mapping[0]

    def translate_segments(
        self,
        segments: Sequence[SourceSegment | dict[str, Any] | str],
        *,
        context_source_segments: Sequence[SourceSegment] | None = None,
        source_language: str | None = None,
        target_language: str | None = None,
        progress_callback: ProgressCallback | None = None,
        cancel_token: CancellationTokenLike | None = None,
    ) -> tuple[TranslatedSegment, ...]:
        """Pure in-memory API for translating a sequence of segments with batching and context."""
        started_at = time.monotonic()
        src_lang = source_language or self._config.source_language
        tgt_lang = target_language or self._config.target_language

        normalized = self._normalize_input_segments(segments)
        if not normalized:
            return ()

        full_source = (
            tuple(context_source_segments)
            if context_source_segments is not None
            else normalized
        )
        source_by_index = {seg.index: seg for seg in full_source}
        total_source_count = max(source_by_index.keys()) + 1 if source_by_index else len(normalized)

        check_cancelled(cancel_token)
        self.load_backend(
            progress_callback=progress_callback,
            cancel_token=cancel_token,
            _started_at=started_at,
        )

        translated = self._run_batches(
            selected=normalized,
            source_by_index=source_by_index,
            total_source_count=total_source_count,
            source_language=src_lang,
            target_language=tgt_lang,
            started_at=started_at,
            progress_callback=progress_callback,
            cancel_token=cancel_token,
        )
        return tuple(translated)

    def translate_file(
        self,
        request: TranslationRequest,
        *,
        progress_callback: ProgressCallback | None = None,
        cancel_token: CancellationTokenLike | None = None,
        transcription_dir: Path | None = None,
    ) -> TranslationResult:
        """Translate a Phase 2 ASR transcript JSON file and save to an independent translation JSON."""
        started_at = time.monotonic()
        target_language = (
            request.target_language or self._config.target_language
        ).strip() or self._config.target_language

        try:
            check_cancelled(cancel_token)

            # Stage 1: Load and validate ASR transcript & selection
            self._emit_progress(
                progress_callback,
                TranslationProgress(
                    stage=TranslationStage.VALIDATING,
                    backend=self._backend.name,
                    model_name=self._backend.model_name,
                    elapsed_seconds=round(time.monotonic() - started_at, 2),
                    message=f"正在讀取逐字稿並驗證選取範圍：{Path(request.transcript_path).name}",
                ),
            )
            transcript = load_asr_transcript(
                request.transcript_path,
                transcription_dir=transcription_dir,
            )
            source_language = transcript.language or self._config.source_language
            selection_mode, selected_segments = select_segments(
                transcript.segments,
                segments_spec=request.segments_spec,
                segment_indices=request.segment_indices,
                start_time=request.start_time,
                end_time=request.end_time,
            )

            resolved_output = (
                Path(request.output_path).expanduser().resolve()
                if request.output_path is not None
                else default_translation_output_path(
                    transcript.source_path,
                    target_language=target_language,
                )
            )

            segments_to_translate = selected_segments
            if (
                request.existing_policy == ExistingTranslationPolicy.SKIP_EXISTING
                and resolved_output.exists()
            ):
                existing = load_translation_json(resolved_output)
                already_done = {seg.index for seg in existing.segments}
                segments_to_translate = tuple(
                    seg for seg in selected_segments if seg.index not in already_done
                )

            check_cancelled(cancel_token)

            # Stage 2: Load backend & translate batches
            translated_segments: list[TranslatedSegment] = []
            if segments_to_translate:
                self.load_backend(
                    progress_callback=progress_callback,
                    cancel_token=cancel_token,
                    _started_at=started_at,
                )
                source_by_index = {seg.index: seg for seg in transcript.segments}
                translated_segments = self._run_batches(
                    selected=segments_to_translate,
                    source_by_index=source_by_index,
                    total_source_count=len(transcript.segments),
                    source_language=source_language,
                    target_language=target_language,
                    started_at=started_at,
                    progress_callback=progress_callback,
                    cancel_token=cancel_token,
                )

            check_cancelled(cancel_token)

            elapsed = round(time.monotonic() - started_at, 2)
            metadata = TranslationMetadata(
                source_transcript=transcript.source_path.name,
                backend=self._backend.name,
                translation_model=self._backend.model_name,
                source_language=source_language,
                target_language=target_language,
                selection_mode=selection_mode.value,
                translated_at=datetime.now(timezone.utc)
                .astimezone()
                .isoformat(timespec="seconds"),
                elapsed_seconds=elapsed,
                device=self._backend.device_info,
                total_source_segments=len(transcript.segments),
                translated_segments_count=len(translated_segments),
            )
            preliminary = TranslationResult(
                metadata=metadata,
                segments=tuple(translated_segments),
                output_path=None,
            )

            # Stage 3: Save / merge translation JSON (never touches Phase 2 ASR JSON)
            self._emit_progress(
                progress_callback,
                TranslationProgress(
                    stage=TranslationStage.SAVING,
                    backend=self._backend.name,
                    model_name=self._backend.model_name,
                    completed_segments=len(translated_segments),
                    total_segments=len(segments_to_translate),
                    elapsed_seconds=round(time.monotonic() - started_at, 2),
                    message=f"正在儲存翻譯結果至：{resolved_output}",
                ),
            )
            saved_result = save_translation_json(
                preliminary,
                resolved_output,
                source_transcript_path=transcript.source_path,
                existing_policy=request.existing_policy,
                source_segments=transcript.segments,
            )

            # Stage 4: Completed
            done_msg = (
                f"翻譯完成，本次翻譯 {len(translated_segments)} 句"
                f"（檔案共含 {len(saved_result.segments)} / {len(transcript.segments)} 句，"
                f"耗時 {saved_result.metadata.elapsed_seconds:.2f}s）。"
            )
            self._emit_progress(
                progress_callback,
                TranslationProgress(
                    stage=TranslationStage.COMPLETED,
                    backend=self._backend.name,
                    model_name=self._backend.model_name,
                    completed_segments=len(translated_segments),
                    total_segments=len(segments_to_translate),
                    elapsed_seconds=saved_result.metadata.elapsed_seconds,
                    message=done_msg,
                ),
            )
            return saved_result

        except TranslationCancelledError as exc:
            self._emit_progress(
                progress_callback,
                TranslationProgress(
                    stage=TranslationStage.CANCELLED,
                    backend=self._backend.name,
                    model_name=self._backend.model_name,
                    elapsed_seconds=round(time.monotonic() - started_at, 2),
                    message=str(exc),
                ),
            )
            raise
        except KeyboardInterrupt as exc:
            cancelled = TranslationCancelledError("翻譯已被使用者中斷。")
            self._emit_progress(
                progress_callback,
                TranslationProgress(
                    stage=TranslationStage.CANCELLED,
                    backend=self._backend.name,
                    model_name=self._backend.model_name,
                    elapsed_seconds=round(time.monotonic() - started_at, 2),
                    message=str(cancelled),
                ),
            )
            raise cancelled from exc
        except TranslationError as exc:
            self._emit_progress(
                progress_callback,
                TranslationProgress(
                    stage=TranslationStage.ERROR,
                    backend=self._backend.name,
                    model_name=self._backend.model_name,
                    elapsed_seconds=round(time.monotonic() - started_at, 2),
                    message=str(exc),
                ),
            )
            raise
        except Exception as exc:
            wrapped = TranslationError(f"翻譯過程發生未預期錯誤：{exc}")
            self._emit_progress(
                progress_callback,
                TranslationProgress(
                    stage=TranslationStage.ERROR,
                    backend=self._backend.name,
                    model_name=self._backend.model_name,
                    elapsed_seconds=round(time.monotonic() - started_at, 2),
                    message=str(wrapped),
                ),
            )
            raise wrapped from exc

    def _run_batches(
        self,
        *,
        selected: Sequence[SourceSegment],
        source_by_index: dict[int, SourceSegment],
        total_source_count: int,
        source_language: str,
        target_language: str,
        started_at: float,
        progress_callback: ProgressCallback | None,
        cancel_token: CancellationTokenLike | None,
    ) -> list[TranslatedSegment]:
        batches = _split_into_contiguous_batches(selected, self._config.batch_size)
        total_batches = len(batches)
        total_segments = len(selected)
        completed_segments = 0
        translated_all: list[TranslatedSegment] = []

        for batch_num, batch in enumerate(batches, start=1):
            check_cancelled(cancel_token)
            self._emit_progress(
                progress_callback,
                TranslationProgress(
                    stage=TranslationStage.TRANSLATING,
                    backend=self._backend.name,
                    model_name=self._backend.model_name,
                    batch_index=batch_num,
                    total_batches=total_batches,
                    completed_segments=completed_segments,
                    total_segments=total_segments,
                    elapsed_seconds=round(time.monotonic() - started_at, 2),
                    message=(
                        f"正在翻譯批次 {batch_num}/{total_batches}"
                        f"（第 {batch[0].index + 1}~{batch[-1].index + 1} 句，共 {len(batch)} 句）..."
                    ),
                ),
            )

            mapping = self._translate_batch_with_recovery(
                batch=batch,
                all_segments_by_index=source_by_index,
                total_source_count=total_source_count,
                source_language=source_language,
                target_language=target_language,
                cancel_token=cancel_token,
            )

            for seg in batch:
                translated_all.append(
                    TranslatedSegment(
                        index=seg.index,
                        start=seg.start,
                        end=seg.end,
                        source_text=seg.text,
                        translated_text=mapping[seg.index],
                    )
                )

            completed_segments += len(batch)
            self._emit_progress(
                progress_callback,
                TranslationProgress(
                    stage=TranslationStage.TRANSLATING,
                    backend=self._backend.name,
                    model_name=self._backend.model_name,
                    batch_index=batch_num,
                    total_batches=total_batches,
                    completed_segments=completed_segments,
                    total_segments=total_segments,
                    elapsed_seconds=round(time.monotonic() - started_at, 2),
                    message=(
                        f"已完成批次 {batch_num}/{total_batches}"
                        f"（累計 {completed_segments}/{total_segments} 句）"
                    ),
                ),
            )

        return translated_all

    def _build_batch_request(
        self,
        batch: Sequence[SourceSegment],
        *,
        all_segments_by_index: dict[int, SourceSegment],
        total_source_count: int,
        source_language: str,
        target_language: str,
    ) -> BackendBatchRequest:
        items = tuple(
            BackendBatchItem(index=seg.index, text=seg.text) for seg in batch
        )
        batch_indices = {seg.index for seg in batch}

        context_before: list[BackendBatchItem] = []
        context_after: list[BackendBatchItem] = []

        if self._backend.supports_context and batch:
            first_idx = batch[0].index
            last_idx = batch[-1].index

            ctx_before_n = max(0, self._config.context_before)
            for idx in range(max(0, first_idx - ctx_before_n), first_idx):
                if idx not in batch_indices and idx in all_segments_by_index:
                    seg = all_segments_by_index[idx]
                    if seg.text.strip():
                        context_before.append(
                            BackendBatchItem(index=seg.index, text=seg.text)
                        )

            ctx_after_n = max(0, self._config.context_after)
            for idx in range(
                last_idx + 1, min(total_source_count, last_idx + 1 + ctx_after_n)
            ):
                if idx not in batch_indices and idx in all_segments_by_index:
                    seg = all_segments_by_index[idx]
                    if seg.text.strip():
                        context_after.append(
                            BackendBatchItem(index=seg.index, text=seg.text)
                        )

        return BackendBatchRequest(
            items=items,
            context_before=tuple(context_before),
            context_after=tuple(context_after),
            source_language=source_language,
            target_language=target_language,
        )

    def _translate_batch_with_recovery(
        self,
        *,
        batch: Sequence[SourceSegment],
        all_segments_by_index: dict[int, SourceSegment],
        total_source_count: int,
        source_language: str,
        target_language: str,
        cancel_token: CancellationTokenLike | None,
    ) -> dict[int, str]:
        """Translate a batch with strict alignment verification and shrink-to-single recovery.

        Recovery ladder on `AlignmentError`:
        1. Re-parse fallback happens inside the backend parser.
        2. If `len(batch) > 1`, split the batch in half and translate sub-batches recursively.
        3. If `len(batch) == 1`, retry up to `single_item_retries` times before raising.
        """
        check_cancelled(cancel_token)
        if not batch:
            return {}

        # Short-circuit batches where every segment is blank/whitespace.
        if all(not seg.text.strip() for seg in batch):
            return {seg.index: "" for seg in batch}

        request = self._build_batch_request(
            batch,
            all_segments_by_index=all_segments_by_index,
            total_source_count=total_source_count,
            source_language=source_language,
            target_language=target_language,
        )

        if len(batch) > 1:
            try:
                mapping = self._backend.translate_batch(
                    request, cancel_token=cancel_token
                )
                _validate_batch_mapping(batch, mapping)
                return mapping
            except AlignmentError:
                # Shrink batch by splitting into two halves and retrying each.
                mid = len(batch) // 2
                left = self._translate_batch_with_recovery(
                    batch=batch[:mid],
                    all_segments_by_index=all_segments_by_index,
                    total_source_count=total_source_count,
                    source_language=source_language,
                    target_language=target_language,
                    cancel_token=cancel_token,
                )
                right = self._translate_batch_with_recovery(
                    batch=batch[mid:],
                    all_segments_by_index=all_segments_by_index,
                    total_source_count=total_source_count,
                    source_language=source_language,
                    target_language=target_language,
                    cancel_token=cancel_token,
                )
                merged = dict(left)
                merged.update(right)
                return merged

        # Single-sentence retry loop
        last_err: AlignmentError | None = None
        for _ in range(self._single_item_retries + 1):
            check_cancelled(cancel_token)
            try:
                mapping = self._backend.translate_batch(
                    request, cancel_token=cancel_token
                )
                _validate_batch_mapping(batch, mapping)
                return mapping
            except AlignmentError as exc:
                last_err = exc
                continue

        assert last_err is not None
        raise AlignmentError(
            f"單句重試 {self._single_item_retries + 1} 次後仍無法對齊"
            f"（第 {batch[0].index + 1} 句）：{last_err}"
        ) from last_err

    @staticmethod
    def _normalize_input_segments(
        segments: Sequence[SourceSegment | dict[str, Any] | str],
    ) -> tuple[SourceSegment, ...]:
        normalized: list[SourceSegment] = []
        for idx, item in enumerate(segments):
            if isinstance(item, SourceSegment):
                normalized.append(item)
            elif isinstance(item, str):
                normalized.append(
                    SourceSegment(index=idx, start=0.0, end=0.0, text=item)
                )
            elif isinstance(item, dict):
                normalized.append(
                    SourceSegment(
                        index=int(item.get("index", idx)),
                        start=float(item.get("start", 0.0)),
                        end=float(item.get("end", 0.0)),
                        text=str(item.get("text", item.get("source_text", ""))),
                    )
                )
            else:
                raise TranslationError(
                    f"不支援的片段輸入型別：{type(item).__name__}"
                )
        return tuple(normalized)

    def _emit_progress(
        self,
        callback: ProgressCallback | None,
        progress: TranslationProgress,
    ) -> None:
        self._current_stage = progress.stage
        if callback is None:
            return
        try:
            callback(progress)
        except Exception:
            return
