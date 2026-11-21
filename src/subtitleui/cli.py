"""CLI entry point for SubtitleUI (Phase 1 downloader + Phase 2 local ASR + Phase 3 translator)."""

from __future__ import annotations

import argparse
from dataclasses import replace
import sys
from pathlib import Path
from typing import Callable, Sequence, TextIO

from .asr import (
    ASRError,
    CancellationToken,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODELS_DIR,
    DEFAULT_TASK,
    ModelConfig,
    SUPPORTED_DEVICES,
    SUPPORTED_MODELS,
    TranscriptionCancelledError,
    TranscriptionEngine,
    TranscriptionProgress,
    TranscriptionRequest,
    TranscriptionStage,
    format_timestamp,
)
from .downloader import (
    DownloaderError,
    DownloadInterruptedError,
    DownloadMode,
    DownloadProgress,
    DownloadRequest,
    DownloadStatus,
    ExistingFilePolicy,
    MediaDownloader,
)
from .paths import (
    DEFAULT_CONFIG_PATH,
    DOWNLOAD_DIR,
    TRANSCRIPTION_DIR,
    default_transcript_output_path,
    default_translation_output_path,
    resolve_media_input_path,
    resolve_transcript_input_path,
)
from .translator import (
    ConfigManager,
    ExistingTranslationPolicy,
    SUPPORTED_PROVIDERS,
    TranslationCancelledError,
    TranslationConfig,
    TranslationEngine,
    TranslationError,
    TranslationProgress,
    TranslationRequest,
    TranslationStage,
)

EngineFactory = Callable[[ModelConfig], TranscriptionEngine]
TranslationEngineFactory = Callable[[TranslationConfig], TranslationEngine]


def format_bytes(num_bytes: int | float | None) -> str:
    """Format byte counts into a compact human-readable string."""
    if num_bytes is None or num_bytes < 0:
        return "未知"
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{value:.1f} TB"


def format_eta(seconds: int | float | None) -> str:
    """Format ETA in seconds as MM:SS or HH:MM:SS."""
    if seconds is None or seconds < 0:
        return "--:--"
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class ConsoleProgressReporter:
    """Renders Phase 1 download progress snapshots to a text stream."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream or sys.stdout
        self._inline_active = False

    def __call__(self, progress: DownloadProgress) -> None:
        if progress.status == DownloadStatus.DOWNLOADING:
            pct = (
                f"{progress.percentage:5.1f}%"
                if progress.percentage is not None
                else "  --.-%"
            )
            downloaded = format_bytes(progress.downloaded_bytes)
            total = (
                format_bytes(progress.total_bytes)
                if progress.total_bytes is not None
                else "未知"
            )
            speed = (
                f"{format_bytes(progress.speed)}/s"
                if progress.speed is not None
                else "--/s"
            )
            eta = format_eta(progress.eta)
            line = f"\r[下載中] {pct} | {downloaded} / {total} | 速度: {speed} | 剩餘: {eta}   "
            self._stream.write(line)
            self._stream.flush()
            self._inline_active = True
            return

        if self._inline_active:
            self._stream.write("\n")
            self._inline_active = False

        prefix_map = {
            DownloadStatus.PREPARING: "[準備中]",
            DownloadStatus.POSTPROCESSING: "[處理中]",
            DownloadStatus.FINISHED: "[完成]",
            DownloadStatus.SKIPPED: "[略過]",
            DownloadStatus.ERROR: "[錯誤]",
        }
        prefix = prefix_map.get(progress.status, "[狀態]")
        if progress.message:
            self._stream.write(f"{prefix} {progress.message}\n")
            self._stream.flush()


class ConsoleASRProgressReporter:
    """Renders Phase 2 ASR workflow stages and elapsed time to a text stream."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream or sys.stdout

    def __call__(self, progress: TranscriptionProgress) -> None:
        stage_labels = {
            TranscriptionStage.VALIDATING: "[檢查中]",
            TranscriptionStage.EXTRACTING_AUDIO: "[擷取音訊]",
            TranscriptionStage.LOADING_MODEL: "[載入模型]",
            TranscriptionStage.TRANSCRIBING: "[辨識中]",
            TranscriptionStage.SAVING: "[儲存中]",
            TranscriptionStage.COMPLETED: "[完成]",
            TranscriptionStage.CANCELLED: "[已取消]",
            TranscriptionStage.ERROR: "[錯誤]",
        }
        label = stage_labels.get(progress.stage, "[狀態]")
        elapsed = f"{progress.elapsed_seconds:.1f}s"
        hw_info = (
            f" | {progress.device.upper()}/{progress.compute_type}"
            if progress.device and progress.compute_type
            else ""
        )
        self._stream.write(
            f"{label} [模型: {progress.model_name}{hw_info} | 耗時: {elapsed}] {progress.message}\n"
        )
        self._stream.flush()


class ConsoleTranslationProgressReporter:
    """Renders Phase 3 translation progress snapshots to a text stream."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream or sys.stdout

    def __call__(self, progress: TranslationProgress) -> None:
        stage_labels = {
            TranslationStage.VALIDATING: "[檢查中]",
            TranslationStage.LOADING_MODEL: "[載入模型]",
            TranslationStage.TRANSLATING: "[翻譯中]",
            TranslationStage.SAVING: "[儲存中]",
            TranslationStage.COMPLETED: "[完成]",
            TranslationStage.CANCELLED: "[已取消]",
            TranslationStage.ERROR: "[錯誤]",
        }
        label = stage_labels.get(progress.stage, "[狀態]")
        elapsed = f"{progress.elapsed_seconds:.1f}s"
        batch_info = (
            f" | 批次 {progress.batch_index}/{progress.total_batches}"
            if progress.total_batches > 0
            else ""
        )
        self._stream.write(
            f"{label} [後端: {progress.backend} | 模型: {progress.model_name}"
            f"{batch_info} | 耗時: {elapsed}] {progress.message}\n"
        )
        self._stream.flush()


def build_download_parser(*, prog: str = "subtitleui") -> argparse.ArgumentParser:
    """Build the Phase 1 download CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "SubtitleUI - 輕量化本機韓文影片下載、語音辨識與翻譯工具\n"
            "  下載模式（Phase 1）：subtitleui <url> [options] 或 subtitleui download <url> [options]\n"
            "  辨識模式（Phase 2）：subtitleui transcribe <audio_or_video> [options]\n"
            "  翻譯模式（Phase 3）：subtitleui translate <transcript.json> [options]"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "url",
        nargs="?",
        help="影片網址（若未提供，程式會以互動方式提示輸入）",
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=[mode.value for mode in DownloadMode],
        default=None,
        help="下載模式：video（影片）或 audio（音訊），預設為 video",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="輸出資料夾路徑，預設為 downloads/",
    )
    parser.add_argument(
        "--audio-format",
        default="m4a",
        help="音訊模式輸出格式（例如 m4a、mp3、wav），預設為 m4a",
    )
    parser.add_argument(
        "--on-exists",
        choices=[policy.value for policy in ExistingFilePolicy],
        default=ExistingFilePolicy.SKIP.value,
        help="目標檔案已存在時的處理方式：skip（略過）、overwrite（覆寫）、fail（報錯），預設為 skip",
    )
    return parser


def build_transcribe_parser() -> argparse.ArgumentParser:
    """Build the Phase 2 local ASR transcription CLI argument parser."""
    models_hint = ", ".join(SUPPORTED_MODELS)
    parser = argparse.ArgumentParser(
        prog="subtitleui transcribe",
        description="SubtitleUI (Phase 2) - 本機 Faster-Whisper 韓文語音辨識（支援整部影片與自訂時間範圍）",
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        help="本機音訊或影片檔案路徑（預設來源為 downloads/；若未提供，將以互動方式提示輸入）",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"Whisper 模型名稱或本機模型資料夾（常用：{models_hint}），預設為 {DEFAULT_MODEL_NAME}",
    )
    parser.add_argument(
        "--language",
        default=DEFAULT_LANGUAGE,
        help=f"辨識語言代碼，預設為 {DEFAULT_LANGUAGE}",
    )
    parser.add_argument(
        "--task",
        choices=["transcribe", "translate"],
        default=DEFAULT_TASK,
        help=f"辨識任務：transcribe（轉錄原文）或 translate，預設為 {DEFAULT_TASK}",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="自訂辨識開始時間（例如 00:12:30 或秒數 750）；未指定則從 00:00:00 開始",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="自訂辨識結束時間（例如 00:18:00 或秒數 1080）；未指定則辨識至影片結尾",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="輸出 JSON 逐字稿路徑（預設為 outputs/transcription/<檔名>.transcript.json）",
    )
    parser.add_argument(
        "--device",
        choices=list(SUPPORTED_DEVICES),
        default="auto",
        help="運算裝置：auto（優先 CUDA+FP16，失敗自動切換 CPU+INT8）、cuda 或 cpu，預設為 auto",
    )
    parser.add_argument(
        "--compute-type",
        default=None,
        help="自訂 CTranslate2 計算精度（如 float16、int8）；預設依裝置自動選擇",
    )
    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODELS_DIR),
        help=f"本機模型快取資料夾路徑，預設為 {DEFAULT_MODELS_DIR}",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam search 寬度，預設為 5",
    )
    parser.add_argument(
        "--vad-filter",
        action="store_true",
        help="啟用 Silero VAD 過濾靜音片段（預設關閉）",
    )
    return parser


def build_translate_parser() -> argparse.ArgumentParser:
    """Build the Phase 3 translation CLI argument parser."""
    providers_hint = ", ".join(SUPPORTED_PROVIDERS)
    parser = argparse.ArgumentParser(
        prog="subtitleui translate",
        description="SubtitleUI (Phase 3) - 韓文逐字稿翻譯為台灣繁體中文（支援 CT2-NLLB 與 Local LLM）",
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        help="Phase 2 逐字稿 JSON 檔案路徑（預設搜尋 outputs/transcription/；未提供時採互動輸入）",
    )
    parser.add_argument(
        "--segments",
        default=None,
        help="指定翻譯句子編號（1-based，例如 1,3,5-10）",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="指定翻譯時間區間開始時間（例如 00:00:10 或秒數 10）",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="指定翻譯時間區間結束時間（例如 00:00:45 或秒數 45）",
    )
    parser.add_argument(
        "--backend",
        choices=list(SUPPORTED_PROVIDERS),
        default=None,
        help=f"指定翻譯後端（{providers_hint}）；未指定時使用 config/config.yaml 設定",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="覆蓋目前後端使用的模型名稱（Local LLM 模型名或 CT2-NLLB repo）",
    )
    parser.add_argument(
        "--target-lang",
        default=None,
        help="目標翻譯語系代碼（預設依 config.yaml，通常為 zh-TW）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="覆蓋每批翻譯句子數（預設依 config.yaml）",
    )
    parser.add_argument(
        "--config",
        default=None,
        help=f"設定檔路徑（預設為 {DEFAULT_CONFIG_PATH}）",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="輸出翻譯 JSON 路徑（預設為 outputs/translation/<stem>.zh-TW.translation.json）",
    )
    parser.add_argument(
        "--on-exists",
        choices=[policy.value for policy in ExistingTranslationPolicy],
        default=ExistingTranslationPolicy.MERGE.value,
        help="翻譯檔已存在時的處理規則：merge（合併更新，預設）、overwrite（覆寫）、skip_existing（略過已翻句子並補翻）",
    )
    return parser


def build_parser() -> argparse.ArgumentParser:
    """Backwards-compatible alias for the default Phase 1 download parser."""
    return build_download_parser()


def _prompt_interactive_download(
    args: argparse.Namespace,
    input_func=input,
    output_stream: TextIO | None = None,
) -> tuple[str, DownloadMode, Path]:
    stream = output_stream or sys.stdout

    url = args.url
    if not url:
        stream.write("請輸入影片網址: ")
        stream.flush()
        url = input_func().strip()

    mode_str = args.mode
    if not mode_str:
        if args.url:
            mode_str = DownloadMode.VIDEO.value
        else:
            stream.write("請選擇下載模式 [video/audio] (預設 video): ")
            stream.flush()
            entered_mode = input_func().strip().lower()
            mode_str = entered_mode or DownloadMode.VIDEO.value

    try:
        mode = DownloadMode(mode_str)
    except ValueError as exc:
        raise DownloaderError(
            f"不支援的下載模式「{mode_str}」，請輸入 video 或 audio。"
        ) from exc

    output_str = args.output
    if output_str is None:
        if args.url:
            output_dir = DOWNLOAD_DIR
        else:
            stream.write(f"請輸入輸出資料夾路徑 (預設 {DOWNLOAD_DIR}): ")
            stream.flush()
            entered_output = input_func().strip()
            output_dir = Path(entered_output) if entered_output else DOWNLOAD_DIR
    else:
        output_dir = Path(output_str)

    return url, mode, output_dir


def _default_transcript_output_path(input_path: Path) -> Path:
    return default_transcript_output_path(input_path)


def _prompt_interactive_transcribe(
    args: argparse.Namespace,
    input_func=input,
    output_stream: TextIO | None = None,
) -> tuple[Path, str, str | None, str | None, Path]:
    stream = output_stream or sys.stdout

    raw_input = args.input_path
    interactive = not raw_input
    if not raw_input:
        stream.write(f"請輸入音訊或影片檔案路徑 (預設來源 {DOWNLOAD_DIR}): ")
        stream.flush()
        raw_input = input_func().strip().strip('"').strip("'")

    input_path = resolve_media_input_path(raw_input) if raw_input else Path(raw_input)

    model_name = args.model
    if not model_name:
        if not interactive:
            model_name = DEFAULT_MODEL_NAME
        else:
            models_list = "/".join(SUPPORTED_MODELS)
            stream.write(f"請選擇 Whisper 模型 [{models_list}] (預設 {DEFAULT_MODEL_NAME}): ")
            stream.flush()
            entered_model = input_func().strip()
            model_name = entered_model or DEFAULT_MODEL_NAME

    start_time = args.start
    end_time = args.end
    if interactive and start_time is None and end_time is None:
        stream.write("請選擇辨識範圍 [1: 整部影片 / 2: 自訂時間範圍] (預設 1): ")
        stream.flush()
        choice = input_func().strip()
        if choice == "2":
            stream.write("請輸入開始時間 (格式 HH:MM:SS，例如 00:12:30): ")
            stream.flush()
            entered_start = input_func().strip()
            start_time = entered_start or None

            stream.write("請輸入結束時間 (格式 HH:MM:SS，例如 00:18:00): ")
            stream.flush()
            entered_end = input_func().strip()
            end_time = entered_end or None

    output_str = args.output
    default_out = _default_transcript_output_path(input_path)
    if output_str is None:
        if not interactive:
            output_path = default_out
        else:
            stream.write(f"請輸入輸出 JSON 路徑 (預設 {default_out}): ")
            stream.flush()
            entered_output = input_func().strip().strip('"').strip("'")
            output_path = Path(entered_output) if entered_output else default_out
    else:
        output_path = Path(output_str)

    return input_path, model_name, start_time, end_time, output_path


def _run_download(
    argv: list[str],
    *,
    downloader: MediaDownloader | None,
    input_func,
    out: TextIO,
    err: TextIO,
) -> int:
    parser = build_download_parser()
    args = parser.parse_args(argv)

    try:
        url, mode, output_dir = _prompt_interactive_download(
            args, input_func=input_func, output_stream=out
        )
        policy = ExistingFilePolicy(args.on_exists)
        request = DownloadRequest(
            url=url,
            mode=mode,
            output_dir=output_dir,
            existing_file_policy=policy,
            audio_format=args.audio_format,
        )
        service = downloader or MediaDownloader()
        reporter = ConsoleProgressReporter(stream=out)
        result = service.download(request, progress_callback=reporter)
    except DownloadInterruptedError as exc:
        err.write(f"下載中斷：{exc}\n")
        return 130
    except KeyboardInterrupt:
        err.write("\n已取消操作。\n")
        return 130
    except DownloaderError as exc:
        err.write(f"下載失敗：{exc}\n")
        return 1
    except Exception as exc:
        err.write(f"發生未預期的錯誤：{exc}\n")
        return 1

    status_label = "已存在（略過下載）" if result.already_exists else "下載成功"
    out.write(f"\n狀態：{status_label}\n")
    out.write(f"標題：{result.title}\n")
    out.write(f"模式：{result.mode.value}\n")
    out.write(f"檔案：{result.file_path}\n")
    out.write(f"大小：{format_bytes(result.file_size)}\n")
    return 0


def _run_transcribe(
    argv: list[str],
    *,
    engine_factory: EngineFactory | None,
    cancel_token: CancellationToken | None,
    input_func,
    out: TextIO,
    err: TextIO,
) -> int:
    parser = build_transcribe_parser()
    args = parser.parse_args(argv)

    try:
        input_path, model_name, start_time, end_time, output_path = (
            _prompt_interactive_transcribe(
                args, input_func=input_func, output_stream=out
            )
        )
        model_config = ModelConfig(
            model_name=model_name,
            device=args.device,
            compute_type=args.compute_type,
            download_root=Path(args.model_dir),
        )
        engine = (
            engine_factory(model_config)
            if engine_factory is not None
            else TranscriptionEngine(model_config)
        )
        request = TranscriptionRequest(
            audio_path=input_path,
            language=args.language,
            task=args.task,
            start_time=start_time,
            end_time=end_time,
            output_path=output_path,
            beam_size=args.beam_size,
            vad_filter=args.vad_filter,
        )
        reporter = ConsoleASRProgressReporter(stream=out)
        result = engine.transcribe(
            request,
            progress_callback=reporter,
            cancel_token=cancel_token,
        )
    except TranscriptionCancelledError as exc:
        err.write(f"辨識已取消：{exc}\n")
        return 130
    except KeyboardInterrupt:
        err.write("\n已取消辨識操作。\n")
        return 130
    except ASRError as exc:
        err.write(f"辨識失敗：{exc}\n")
        return 1
    except Exception as exc:
        err.write(f"發生未預期的錯誤：{exc}\n")
        return 1

    meta = result.metadata
    out.write("\n辨識摘要：\n")
    out.write(f"來源：{meta.source}\n")
    out.write(f"模型：{meta.model}（{meta.device.upper()} / {meta.compute_type}）\n")
    out.write(f"語言：{meta.language}（任務：{meta.task}）\n")
    if meta.language_probability is not None:
        out.write(f"語言信心度：{meta.language_probability:.4f}\n")
    out.write(
        f"辨識區間：{format_timestamp(meta.range_start)} ~ {format_timestamp(meta.range_end)}"
        f"（原片長度：{format_timestamp(meta.source_duration)}）\n"
    )
    out.write(f"片段數量：{len(result.segments)}\n")
    if meta.elapsed_seconds is not None:
        out.write(f"總耗時：{meta.elapsed_seconds:.2f} 秒\n")
    if result.output_path is not None:
        out.write(f"JSON 輸出：{result.output_path}\n")

    if result.segments:
        preview_count = min(5, len(result.segments))
        out.write(f"\n前 {preview_count} 段預覽（原影片時間戳記）：\n")
        for seg in result.segments[:preview_count]:
            out.write(
                f"  [{format_timestamp(seg.start, include_millis=True)} -> "
                f"{format_timestamp(seg.end, include_millis=True)}] {seg.text}\n"
            )
    else:
        out.write("\n（此時間範圍內未辨識出有效語音片段。）\n")

    return 0


def _prompt_interactive_translate(
    args: argparse.Namespace,
    config: TranslationConfig,
    input_func=input,
    output_stream: TextIO | None = None,
) -> tuple[Path, str | None, str | None, str | None, Path | None]:
    stream = output_stream or sys.stdout

    raw_input = args.input_path
    interactive = not raw_input
    if not raw_input:
        stream.write(f"請輸入逐字稿 JSON 路徑 (預設搜尋 {TRANSCRIPTION_DIR}): ")
        stream.flush()
        raw_input = input_func().strip().strip('"').strip("'")

    input_path = (
        resolve_transcript_input_path(raw_input) if raw_input else Path(raw_input)
    )

    segments_spec = args.segments
    start_time = args.start
    end_time = args.end
    if interactive and segments_spec is None and start_time is None and end_time is None:
        stream.write("請選擇翻譯範圍 [1: 全文 / 2: 指定句子 / 3: 指定時間區間] (預設 1): ")
        stream.flush()
        choice = input_func().strip()
        if choice == "2":
            stream.write("請輸入句子編號（1-based，例如 1,3,5-10）: ")
            stream.flush()
            entered = input_func().strip()
            segments_spec = entered or None
        elif choice == "3":
            stream.write("請輸入開始時間 (格式 HH:MM:SS，例如 00:00:10): ")
            stream.flush()
            entered_start = input_func().strip()
            start_time = entered_start or None

            stream.write("請輸入結束時間 (格式 HH:MM:SS，例如 00:00:45): ")
            stream.flush()
            entered_end = input_func().strip()
            end_time = entered_end or None

    target_lang = args.target_lang or config.target_language
    output_path: Path | None
    if args.output is not None:
        output_path = Path(args.output)
    elif interactive:
        default_out = default_translation_output_path(
            input_path, target_language=target_lang
        )
        stream.write(f"請輸入翻譯輸出 JSON 路徑 (預設 {default_out}): ")
        stream.flush()
        entered_output = input_func().strip().strip('"').strip("'")
        output_path = Path(entered_output) if entered_output else default_out
    else:
        output_path = None

    return input_path, segments_spec, start_time, end_time, output_path


def _apply_translate_cli_overrides(
    config: TranslationConfig,
    args: argparse.Namespace,
) -> TranslationConfig:
    updated = config
    if args.backend:
        updated = replace(updated, provider=args.backend)
    if args.target_lang:
        updated = replace(updated, target_language=args.target_lang.strip())
    if args.batch_size is not None:
        if args.batch_size < 1:
            raise TranslationError("--batch-size 必須大於或等於 1。")
        updated = replace(updated, batch_size=args.batch_size)
    if args.model:
        model_val = args.model.strip()
        if updated.provider == "local-llm":
            updated = replace(
                updated,
                local_llm=replace(updated.local_llm, model=model_val),
            )
        elif updated.provider == "ct2-nllb":
            updated = replace(
                updated,
                ct2_nllb=replace(updated.ct2_nllb, model_repo=model_val),
            )
    return updated


def _run_translate(
    argv: list[str],
    *,
    translation_engine_factory: TranslationEngineFactory | None,
    cancel_token: CancellationToken | None,
    input_func,
    out: TextIO,
    err: TextIO,
) -> int:
    parser = build_translate_parser()
    args = parser.parse_args(argv)

    engine: TranslationEngine | None = None
    try:
        config_manager = ConfigManager(args.config)
        created_default = config_manager.ensure_config_exists()
        if created_default:
            out.write(f"[設定] 已自動建立預設設定檔：{config_manager.config_path}\n")
            out.flush()

        config = config_manager.load(auto_create=False)
        config = _apply_translate_cli_overrides(config, args)

        input_path, segments_spec, start_time, end_time, output_path = (
            _prompt_interactive_translate(
                args, config, input_func=input_func, output_stream=out
            )
        )

        engine = (
            translation_engine_factory(config)
            if translation_engine_factory is not None
            else TranslationEngine(config)
        )
        request = TranslationRequest(
            transcript_path=input_path,
            segments_spec=segments_spec,
            start_time=start_time,
            end_time=end_time,
            output_path=output_path,
            target_language=config.target_language,
            existing_policy=ExistingTranslationPolicy(args.on_exists),
        )
        reporter = ConsoleTranslationProgressReporter(stream=out)
        result = engine.translate_file(
            request,
            progress_callback=reporter,
            cancel_token=cancel_token,
        )
    except TranslationCancelledError as exc:
        err.write(f"翻譯已取消：{exc}\n")
        return 130
    except KeyboardInterrupt:
        err.write("\n已取消翻譯操作。\n")
        return 130
    except TranslationError as exc:
        err.write(f"翻譯失敗：{exc}\n")
        return 1
    except Exception as exc:
        err.write(f"發生未預期的錯誤：{exc}\n")
        return 1
    finally:
        if engine is not None:
            try:
                engine.unload_backend()
            except Exception:
                pass

    meta = result.metadata
    device_suffix = f"（{meta.device}）" if meta.device else ""
    out.write("\n翻譯摘要：\n")
    out.write(f"來源逐字稿：{meta.source_transcript}\n")
    out.write(f"翻譯後端：{meta.backend} / {meta.translation_model}{device_suffix}\n")
    out.write(f"語言方向：{meta.source_language} -> {meta.target_language}\n")
    out.write(f"選取模式：{meta.selection_mode}\n")
    if meta.total_source_segments is not None:
        out.write(
            f"翻譯片段數：{len(result.segments)} / {meta.total_source_segments}\n"
        )
    else:
        out.write(f"翻譯片段數：{len(result.segments)}\n")
    out.write(f"總耗時：{meta.elapsed_seconds:.2f} 秒\n")
    if result.output_path is not None:
        out.write(f"翻譯輸出：{result.output_path}\n")

    if result.segments:
        preview_count = min(5, len(result.segments))
        out.write(f"\n前 {preview_count} 句預覽：\n")
        for seg in result.segments[:preview_count]:
            out.write(
                f"  #{seg.index + 1} [{format_timestamp(seg.start, include_millis=True)} -> "
                f"{format_timestamp(seg.end, include_millis=True)}]\n"
                f"    原文：{seg.source_text}\n"
                f"    譯文：{seg.translated_text}\n"
            )
    else:
        out.write("\n（所選範圍內沒有可翻譯的片段。）\n")

    return 0


def main(
    argv: Sequence[str] | None = None,
    *,
    downloader: MediaDownloader | None = None,
    engine_factory: EngineFactory | None = None,
    translation_engine_factory: TranslationEngineFactory | None = None,
    cancel_token: CancellationToken | None = None,
    input_func=input,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run the SubtitleUI CLI (download, transcribe, or translate) and return an exit status code."""
    out = stdout or sys.stdout
    err = stderr or sys.stderr
    raw_args = list(argv) if argv is not None else list(sys.argv[1:])

    if raw_args and raw_args[0] == "translate":
        return _run_translate(
            raw_args[1:],
            translation_engine_factory=translation_engine_factory,
            cancel_token=cancel_token,
            input_func=input_func,
            out=out,
            err=err,
        )

    if raw_args and raw_args[0] == "transcribe":
        return _run_transcribe(
            raw_args[1:],
            engine_factory=engine_factory,
            cancel_token=cancel_token,
            input_func=input_func,
            out=out,
            err=err,
        )

    if raw_args and raw_args[0] == "download":
        return _run_download(
            raw_args[1:],
            downloader=downloader,
            input_func=input_func,
            out=out,
            err=err,
        )

    return _run_download(
        raw_args,
        downloader=downloader,
        input_func=input_func,
        out=out,
        err=err,
    )
