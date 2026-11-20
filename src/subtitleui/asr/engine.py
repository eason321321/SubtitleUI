"""Reusable local Faster-Whisper Korean transcription engine."""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .audio import AudioExtractor
from .errors import (
    ASRError,
    ModelLoadError,
    TranscriptionCancelledError,
    TranscriptionError,
)
from .hardware import (
    check_cuda_availability,
    is_cuda_runtime_error,
    register_nvidia_dll_directories,
    resolve_hardware_candidates,
)
from .models import (
    CancellationToken,
    DEFAULT_LANGUAGE,
    DEFAULT_TASK,
    HardwareConfig,
    ModelConfig,
    ProgressCallback,
    SUPPORTED_MODELS,
    TranscriptSegment,
    TranscriptionMetadata,
    TranscriptionProgress,
    TranscriptionRequest,
    TranscriptionResult,
    TranscriptionStage,
)
from .storage import save_transcript_json
from .time_utils import apply_timestamp_offset, format_timestamp, resolve_time_range

ModelFactory = Callable[..., Any]

_MODEL_ALIASES: dict[str, str] = {
    "whisper-small": "small",
    "whisper-medium": "medium",
    "whisper-large-v3": "large-v3",
    "whisper-large-v3-turbo": "large-v3-turbo",
    "large-v3 turbo": "large-v3-turbo",
    "large_v3_turbo": "large-v3-turbo",
    "large_v3": "large-v3",
    "turbo": "large-v3-turbo",
}

_KNOWN_WHISPER_MODELS: set[str] = {
    *SUPPORTED_MODELS,
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small.en",
    "medium.en",
    "large-v1",
    "large-v2",
    "large",
    "distil-small.en",
    "distil-medium.en",
    "distil-large-v2",
    "distil-large-v3",
    "distil-large-v3.5",
}


def normalize_model_name(model_name: str) -> str:
    """Normalize and validate a Whisper model identifier or local directory path."""
    if not isinstance(model_name, str) or not model_name.strip():
        raise ModelLoadError("模型名稱不可為空白。")

    raw = model_name.strip()
    local_path = Path(raw).expanduser()
    if local_path.exists() and local_path.is_dir():
        return str(local_path.resolve())

    key = raw.lower()
    normalized = _MODEL_ALIASES.get(key, key)
    if normalized in _KNOWN_WHISPER_MODELS:
        return normalized

    supported_str = ", ".join(SUPPORTED_MODELS)
    raise ModelLoadError(
        f"不支援的模型名稱「{model_name}」。建議使用：{supported_str}，或指定本機模型資料夾路徑。"
    )


def is_model_cached(model_name: str, download_root: Path) -> bool:
    """Heuristically check whether the requested model already exists locally."""
    direct_path = Path(model_name)
    if direct_path.is_dir() and (direct_path / "model.bin").exists():
        return True

    if not download_root.is_dir():
        return False

    needle = model_name.lower()
    for entry in download_root.iterdir():
        if not entry.is_dir():
            continue
        name_lower = entry.name.lower()
        if f"faster-whisper-{needle}" in name_lower or name_lower.endswith(f"--{needle}"):
            if any(entry.rglob("model.bin")):
                return True
    return False


def _default_model_factory(
    model_name: str,
    *,
    device: str,
    compute_type: str,
    download_root: str,
    cpu_threads: int = 0,
) -> Any:
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    register_nvidia_dll_directories()
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise ModelLoadError(
            "尚未安裝 faster-whisper，請先執行 pip install -r requirements.txt。"
        ) from exc

    kwargs: dict[str, Any] = {
        "device": device,
        "compute_type": compute_type,
        "download_root": download_root,
    }
    if cpu_threads > 0:
        kwargs["cpu_threads"] = cpu_threads
    return WhisperModel(model_name, **kwargs)


class TranscriptionEngine:
    """Reusable local ASR engine separating model loading, audio prep, and inference.

    A single engine instance keeps its loaded Faster-Whisper model in memory so
    subsequent `transcribe()` calls reuse it without reinitialization.
    """

    def __init__(
        self,
        model_config: ModelConfig | None = None,
        *,
        audio_extractor: AudioExtractor | None = None,
        model_factory: ModelFactory | None = None,
        cuda_checker: Callable[[], tuple[bool, str | None]] | None = None,
    ) -> None:
        raw_cfg = model_config or ModelConfig()
        normalized_name = normalize_model_name(raw_cfg.model_name)
        self._model_config = ModelConfig(
            model_name=normalized_name,
            device=raw_cfg.device,
            compute_type=raw_cfg.compute_type,
            download_root=Path(raw_cfg.download_root),
            cpu_threads=raw_cfg.cpu_threads,
        )
        self._audio_extractor = audio_extractor or AudioExtractor()
        self._model_factory = model_factory or _default_model_factory
        self._cuda_checker = cuda_checker or check_cuda_availability
        self._model: Any | None = None
        self._hardware_config: HardwareConfig | None = None
        self._current_stage: TranscriptionStage | None = None

    @property
    def model_config(self) -> ModelConfig:
        """Return the normalized model configuration."""
        return self._model_config

    @property
    def hardware_config(self) -> HardwareConfig | None:
        """Return the active hardware configuration once loaded."""
        return self._hardware_config

    @property
    def is_loaded(self) -> bool:
        """Return True if the Whisper model is already initialized and ready."""
        return self._model is not None and self._hardware_config is not None

    @property
    def current_stage(self) -> TranscriptionStage | None:
        """Return the most recently entered workflow stage."""
        return self._current_stage

    def unload_model(self) -> None:
        """Release the loaded model instance from memory."""
        self._model = None
        self._hardware_config = None

    def load_model(
        self,
        *,
        progress_callback: ProgressCallback | None = None,
        cancel_token: CancellationToken | None = None,
        _started_at: float | None = None,
    ) -> HardwareConfig:
        """Load the configured Whisper model once and cache it on this engine."""
        if self.is_loaded and self._hardware_config is not None:
            return self._hardware_config

        if cancel_token is not None:
            cancel_token.raise_if_cancelled()

        started_at = _started_at if _started_at is not None else time.monotonic()
        download_root = self._model_config.download_root.expanduser().resolve()
        try:
            download_root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ModelLoadError(
                f"無法建立模型快取資料夾「{download_root}」：{exc}"
            ) from exc

        candidates = resolve_hardware_candidates(
            preferred_device=self._model_config.device,
            preferred_compute_type=self._model_config.compute_type,
            cuda_checker=self._cuda_checker,
        )

        cached = is_model_cached(self._model_config.model_name, download_root)
        last_error: Exception | None = None

        for idx, candidate in enumerate(candidates):
            if cancel_token is not None:
                cancel_token.raise_if_cancelled()

            if idx == 0:
                if cached:
                    load_msg = (
                        f"正在從本機快取載入模型「{self._model_config.model_name}」"
                        f"（{candidate.device.upper()} / {candidate.compute_type}）..."
                    )
                else:
                    load_msg = (
                        f"首次使用模型「{self._model_config.model_name}」，將下載並快取至「{download_root}」"
                        f"（{candidate.device.upper()} / {candidate.compute_type}）..."
                    )
                if candidate.fallback_used and candidate.fallback_reason:
                    load_msg = f"{candidate.fallback_reason}。{load_msg}"
            else:
                load_msg = (
                    f"{candidate.fallback_reason or '已切換備援裝置'}，"
                    f"改用 {candidate.device.upper()} / {candidate.compute_type} 載入模型「{self._model_config.model_name}」..."
                )

            self._emit_progress(
                progress_callback,
                TranscriptionProgress(
                    stage=TranscriptionStage.LOADING_MODEL,
                    model_name=self._model_config.model_name,
                    device=candidate.device,
                    compute_type=candidate.compute_type,
                    elapsed_seconds=round(time.monotonic() - started_at, 2),
                    message=load_msg,
                ),
            )

            try:
                model = self._model_factory(
                    self._model_config.model_name,
                    device=candidate.device,
                    compute_type=candidate.compute_type,
                    download_root=str(download_root),
                    cpu_threads=self._model_config.cpu_threads,
                )
            except KeyboardInterrupt as exc:
                cancelled = TranscriptionCancelledError("模型載入已被使用者中斷。")
                self._emit_progress(
                    progress_callback,
                    TranscriptionProgress(
                        stage=TranscriptionStage.CANCELLED,
                        model_name=self._model_config.model_name,
                        elapsed_seconds=round(time.monotonic() - started_at, 2),
                        message=str(cancelled),
                    ),
                )
                raise cancelled from exc
            except Exception as exc:
                last_error = exc
                has_fallback = idx + 1 < len(candidates)
                if has_fallback and candidate.device == "cuda":
                    continue
                err = self._wrap_model_load_error(candidate, exc)
                self._emit_progress(
                    progress_callback,
                    TranscriptionProgress(
                        stage=TranscriptionStage.ERROR,
                        model_name=self._model_config.model_name,
                        device=candidate.device,
                        compute_type=candidate.compute_type,
                        elapsed_seconds=round(time.monotonic() - started_at, 2),
                        message=str(err),
                    ),
                )
                raise err from exc

            self._model = model
            self._hardware_config = candidate
            return candidate

        err = ModelLoadError(
            f"無法載入模型「{self._model_config.model_name}」：{last_error}"
        )
        self._emit_progress(
            progress_callback,
            TranscriptionProgress(
                stage=TranscriptionStage.ERROR,
                model_name=self._model_config.model_name,
                elapsed_seconds=round(time.monotonic() - started_at, 2),
                message=str(err),
            ),
        )
        raise err

    def transcribe(
        self,
        request: TranscriptionRequest,
        *,
        progress_callback: ProgressCallback | None = None,
        cancel_token: CancellationToken | None = None,
    ) -> TranscriptionResult:
        """Run the full transcription workflow for full media or a custom range."""
        started_at = time.monotonic()
        language = (request.language or DEFAULT_LANGUAGE).strip()
        task = (request.task or DEFAULT_TASK).strip().lower()

        if task not in {"transcribe", "translate"}:
            raise TranscriptionError(
                f"不支援的辨識任務「{task}」，請使用 transcribe 或 translate。"
            )
        if not language:
            raise TranscriptionError("辨識語言設定不可為空白（預設為 ko）。")

        try:
            if cancel_token is not None:
                cancel_token.raise_if_cancelled()

            # Stage 1: Validate input media & resolve time range
            self._emit_progress(
                progress_callback,
                TranscriptionProgress(
                    stage=TranscriptionStage.VALIDATING,
                    model_name=self._model_config.model_name,
                    elapsed_seconds=round(time.monotonic() - started_at, 2),
                    message=f"正在檢查音訊來源與時間範圍：{Path(request.audio_path).name}",
                ),
            )
            source_path = Path(request.audio_path).expanduser().resolve()
            source_duration = self._audio_extractor.probe_duration(source_path)
            time_range, is_custom_range = resolve_time_range(
                request.start_time,
                request.end_time,
                source_duration,
            )

            if cancel_token is not None:
                cancel_token.raise_if_cancelled()

            # Stage 2: Extract audio slice (16kHz mono PCM WAV) to auto-cleaned temp dir
            range_desc = (
                f"{format_timestamp(time_range.start)} ~ {format_timestamp(time_range.end)}"
                if is_custom_range
                else f"整部影片（00:00:00 ~ {format_timestamp(source_duration)}）"
            )
            self._emit_progress(
                progress_callback,
                TranscriptionProgress(
                    stage=TranscriptionStage.EXTRACTING_AUDIO,
                    model_name=self._model_config.model_name,
                    elapsed_seconds=round(time.monotonic() - started_at, 2),
                    clip_duration_seconds=time_range.duration,
                    message=f"正在使用 FFmpeg 擷取 16kHz 單聲道 WAV 音訊：{range_desc}",
                ),
            )

            with self._audio_extractor.extract_to_temp(
                source_path,
                time_range=time_range,
                is_custom_range=is_custom_range,
                cancel_token=cancel_token,
            ) as extracted:
                if cancel_token is not None:
                    cancel_token.raise_if_cancelled()

                # Stage 3: Ensure model is loaded (reuses cached model if already loaded)
                hw = self.load_model(
                    progress_callback=progress_callback,
                    cancel_token=cancel_token,
                    _started_at=started_at,
                )

                # Stage 4: Run Faster-Whisper inference & iterate generator
                segments, info, hw = self._run_inference_with_fallback(
                    wav_path=extracted.wav_path,
                    clip_duration=extracted.duration,
                    range_start=time_range.start,
                    language=language,
                    task=task,
                    beam_size=request.beam_size,
                    vad_filter=request.vad_filter,
                    started_at=started_at,
                    progress_callback=progress_callback,
                    cancel_token=cancel_token,
                )

            if cancel_token is not None:
                cancel_token.raise_if_cancelled()

            detected_language = getattr(info, "language", None)
            raw_lang_prob = getattr(info, "language_probability", None)
            language_probability: float | None = None
            if isinstance(raw_lang_prob, (int, float)):
                language_probability = round(float(raw_lang_prob), 4)

            total_elapsed = round(time.monotonic() - started_at, 2)
            metadata = TranscriptionMetadata(
                source=str(source_path),
                model=self._model_config.model_name,
                language=language,
                task=task,
                transcribed_at=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
                range_start=time_range.start,
                range_end=time_range.end,
                source_duration=source_duration,
                device=hw.device,
                compute_type=hw.compute_type,
                detected_language=str(detected_language) if detected_language is not None else None,
                language_probability=language_probability,
                elapsed_seconds=total_elapsed,
            )

            result = TranscriptionResult(
                metadata=metadata,
                segments=tuple(segments),
                output_path=None,
            )

            # Stage 5: Persist JSON if output_path was provided
            if request.output_path is not None:
                self._emit_progress(
                    progress_callback,
                    TranscriptionProgress(
                        stage=TranscriptionStage.SAVING,
                        model_name=self._model_config.model_name,
                        device=hw.device,
                        compute_type=hw.compute_type,
                        elapsed_seconds=round(time.monotonic() - started_at, 2),
                        segments_count=len(result.segments),
                        message=f"正在儲存 JSON 逐字稿至：{request.output_path}",
                    ),
                )
                saved_path = save_transcript_json(result, request.output_path)
                result = TranscriptionResult(
                    metadata=result.metadata,
                    segments=result.segments,
                    output_path=saved_path,
                )

            # Stage 6: Completed
            if result.segments:
                done_msg = (
                    f"辨識完成，共取得 {len(result.segments)} 個語音片段"
                    f"（耗時 {total_elapsed:.1f}s，裝置 {hw.device.upper()} / {hw.compute_type}）。"
                )
            else:
                done_msg = (
                    f"辨識完成，但音訊區間內未偵測到有效語音內容"
                    f"（耗時 {total_elapsed:.1f}s，裝置 {hw.device.upper()} / {hw.compute_type}）。"
                )

            self._emit_progress(
                progress_callback,
                TranscriptionProgress(
                    stage=TranscriptionStage.COMPLETED,
                    model_name=self._model_config.model_name,
                    device=hw.device,
                    compute_type=hw.compute_type,
                    elapsed_seconds=total_elapsed,
                    processed_audio_seconds=time_range.duration,
                    clip_duration_seconds=time_range.duration,
                    segments_count=len(result.segments),
                    message=done_msg,
                ),
            )
            return result

        except TranscriptionCancelledError as exc:
            self._emit_progress(
                progress_callback,
                TranscriptionProgress(
                    stage=TranscriptionStage.CANCELLED,
                    model_name=self._model_config.model_name,
                    device=self._hardware_config.device if self._hardware_config else None,
                    compute_type=self._hardware_config.compute_type if self._hardware_config else None,
                    elapsed_seconds=round(time.monotonic() - started_at, 2),
                    message=str(exc),
                ),
            )
            raise
        except KeyboardInterrupt as exc:
            cancelled = TranscriptionCancelledError("語音辨識已被使用者中斷。")
            self._emit_progress(
                progress_callback,
                TranscriptionProgress(
                    stage=TranscriptionStage.CANCELLED,
                    model_name=self._model_config.model_name,
                    device=self._hardware_config.device if self._hardware_config else None,
                    compute_type=self._hardware_config.compute_type if self._hardware_config else None,
                    elapsed_seconds=round(time.monotonic() - started_at, 2),
                    message=str(cancelled),
                ),
            )
            raise cancelled from exc
        except ASRError as exc:
            self._emit_progress(
                progress_callback,
                TranscriptionProgress(
                    stage=TranscriptionStage.ERROR,
                    model_name=self._model_config.model_name,
                    device=self._hardware_config.device if self._hardware_config else None,
                    compute_type=self._hardware_config.compute_type if self._hardware_config else None,
                    elapsed_seconds=round(time.monotonic() - started_at, 2),
                    message=str(exc),
                ),
            )
            raise
        except Exception as exc:
            wrapped = TranscriptionError(f"語音辨識過程發生未預期錯誤：{exc}")
            self._emit_progress(
                progress_callback,
                TranscriptionProgress(
                    stage=TranscriptionStage.ERROR,
                    model_name=self._model_config.model_name,
                    device=self._hardware_config.device if self._hardware_config else None,
                    compute_type=self._hardware_config.compute_type if self._hardware_config else None,
                    elapsed_seconds=round(time.monotonic() - started_at, 2),
                    message=str(wrapped),
                ),
            )
            raise wrapped from exc

    def _run_inference_with_fallback(
        self,
        *,
        wav_path: Path,
        clip_duration: float,
        range_start: float,
        language: str,
        task: str,
        beam_size: int,
        vad_filter: bool,
        started_at: float,
        progress_callback: ProgressCallback | None,
        cancel_token: CancellationToken | None,
    ) -> tuple[list[TranscriptSegment], Any, HardwareConfig]:
        assert self._model is not None
        assert self._hardware_config is not None

        try:
            segments, info = self._consume_segments(
                model=self._model,
                hw=self._hardware_config,
                wav_path=wav_path,
                clip_duration=clip_duration,
                range_start=range_start,
                language=language,
                task=task,
                beam_size=beam_size,
                vad_filter=vad_filter,
                started_at=started_at,
                progress_callback=progress_callback,
                cancel_token=cancel_token,
            )
            return segments, info, self._hardware_config
        except Exception as exc:
            # Handle lazy CUDA/cuDNN load failures during first inference in auto mode
            can_fallback = (
                self._model_config.device.strip().lower() == "auto"
                and self._hardware_config is not None
                and self._hardware_config.device == "cuda"
                and is_cuda_runtime_error(exc)
            )
            if not can_fallback:
                raise

            self.unload_model()
            fallback_hw = HardwareConfig(
                device="cpu",
                compute_type="int8",
                fallback_used=True,
                fallback_reason=f"CUDA 推論失敗（{exc}），已自動切換為 CPU (INT8)",
            )
            self._emit_progress(
                progress_callback,
                TranscriptionProgress(
                    stage=TranscriptionStage.LOADING_MODEL,
                    model_name=self._model_config.model_name,
                    device=fallback_hw.device,
                    compute_type=fallback_hw.compute_type,
                    elapsed_seconds=round(time.monotonic() - started_at, 2),
                    message=f"{fallback_hw.fallback_reason}，正在重新載入模型...",
                ),
            )
            download_root = self._model_config.download_root.expanduser().resolve()
            try:
                self._model = self._model_factory(
                    self._model_config.model_name,
                    device=fallback_hw.device,
                    compute_type=fallback_hw.compute_type,
                    download_root=str(download_root),
                    cpu_threads=self._model_config.cpu_threads,
                )
            except Exception as fallback_exc:
                raise self._wrap_model_load_error(fallback_hw, fallback_exc) from fallback_exc

            self._hardware_config = fallback_hw
            segments, info = self._consume_segments(
                model=self._model,
                hw=fallback_hw,
                wav_path=wav_path,
                clip_duration=clip_duration,
                range_start=range_start,
                language=language,
                task=task,
                beam_size=beam_size,
                vad_filter=vad_filter,
                started_at=started_at,
                progress_callback=progress_callback,
                cancel_token=cancel_token,
            )
            return segments, info, fallback_hw

    def _consume_segments(
        self,
        *,
        model: Any,
        hw: HardwareConfig,
        wav_path: Path,
        clip_duration: float,
        range_start: float,
        language: str,
        task: str,
        beam_size: int,
        vad_filter: bool,
        started_at: float,
        progress_callback: ProgressCallback | None,
        cancel_token: CancellationToken | None,
    ) -> tuple[list[TranscriptSegment], Any]:
        if cancel_token is not None:
            cancel_token.raise_if_cancelled()

        self._emit_progress(
            progress_callback,
            TranscriptionProgress(
                stage=TranscriptionStage.TRANSCRIBING,
                model_name=self._model_config.model_name,
                device=hw.device,
                compute_type=hw.compute_type,
                elapsed_seconds=round(time.monotonic() - started_at, 2),
                processed_audio_seconds=0.0,
                clip_duration_seconds=clip_duration,
                segments_count=0,
                message=(
                    f"開始辨識韓文語音（模型：{self._model_config.model_name}，"
                    f"裝置：{hw.device.upper()} / {hw.compute_type}）..."
                ),
            ),
        )

        segments_iter, info = model.transcribe(
            str(wav_path),
            language=language,
            task=task,
            beam_size=beam_size,
            vad_filter=vad_filter,
        )

        collected: list[TranscriptSegment] = []
        # Faster-Whisper returns a lazy generator; iterate it to drive recognition.
        for raw_segment in segments_iter:
            if cancel_token is not None:
                cancel_token.raise_if_cancelled()

            seg_text = str(getattr(raw_segment, "text", "")).strip()
            seg_start_rel = max(0.0, float(getattr(raw_segment, "start", 0.0)))
            seg_end_rel = max(seg_start_rel, float(getattr(raw_segment, "end", seg_start_rel)))

            if seg_text:
                orig_start = apply_timestamp_offset(seg_start_rel, range_start)
                orig_end = apply_timestamp_offset(seg_end_rel, range_start)
                collected.append(
                    TranscriptSegment(
                        start=orig_start,
                        end=orig_end,
                        text=seg_text,
                    )
                )

            self._emit_progress(
                progress_callback,
                TranscriptionProgress(
                    stage=TranscriptionStage.TRANSCRIBING,
                    model_name=self._model_config.model_name,
                    device=hw.device,
                    compute_type=hw.compute_type,
                    elapsed_seconds=round(time.monotonic() - started_at, 2),
                    processed_audio_seconds=round(min(seg_end_rel, clip_duration), 2),
                    clip_duration_seconds=clip_duration,
                    segments_count=len(collected),
                    message=(
                        f"辨識中（已處理音訊 {format_timestamp(min(seg_end_rel, clip_duration))} / "
                        f"{format_timestamp(clip_duration)}，片段數：{len(collected)}）"
                    ),
                ),
            )

        return collected, info

    def _wrap_model_load_error(
        self,
        candidate: HardwareConfig,
        exc: Exception,
    ) -> ModelLoadError:
        return ModelLoadError(
            f"載入模型「{self._model_config.model_name}」失敗"
            f"（裝置：{candidate.device}，精度：{candidate.compute_type}）：{exc}"
        )

    def _emit_progress(
        self,
        callback: ProgressCallback | None,
        progress: TranscriptionProgress,
    ) -> None:
        self._current_stage = progress.stage
        if callback is None:
            return
        try:
            callback(progress)
        except Exception:
            # Progress reporting failures must never abort core transcription.
            return
