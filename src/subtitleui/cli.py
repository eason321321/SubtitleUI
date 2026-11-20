"""Minimal CLI entry point for SubtitleUI Phase 1 media downloading."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence, TextIO

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
    """Renders download progress snapshots to a text stream."""

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


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="subtitleui",
        description="SubtitleUI (Phase 1) - 輕量化本機影片／音訊下載工具",
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
        help="輸出資料夾路徑，預設為 downloads",
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


def _prompt_interactive(
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
            output_str = "downloads"
        else:
            stream.write("請輸入輸出資料夾路徑 (預設 downloads): ")
            stream.flush()
            entered_output = input_func().strip()
            output_str = entered_output or "downloads"

    return url, mode, Path(output_str)


def main(
    argv: Sequence[str] | None = None,
    *,
    downloader: MediaDownloader | None = None,
    input_func=input,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run the Phase 1 CLI and return an exit status code."""
    out = stdout or sys.stdout
    err = stderr or sys.stderr
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        url, mode, output_dir = _prompt_interactive(
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
