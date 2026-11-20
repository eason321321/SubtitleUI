"""Core media download service built on yt-dlp Python API."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable
from urllib.error import URLError

import yt_dlp
from yt_dlp.utils import DownloadError, ExtractorError, PostProcessingError

from .errors import (
    DownloaderError,
    DownloadInterruptedError,
    ExternalToolError,
    FileAlreadyExistsError,
    InvalidURLError,
    MediaDownloadError,
    NetworkDownloadError,
)
from .models import (
    DownloadMode,
    DownloadProgress,
    DownloadRequest,
    DownloadResult,
    DownloadStatus,
    ExistingFilePolicy,
    ProgressCallback,
)
from .validators import ensure_output_dir, validate_audio_format, validate_url

_NETWORK_KEYWORDS = (
    "getaddrinfo",
    "name or service not known",
    "temporary failure in name resolution",
    "failed to resolve",
    "timed out",
    "timeout",
    "connection refused",
    "connection reset",
    "connection aborted",
    "network is unreachable",
    "no route to host",
    "urlopen error",
    "unable to download webpage",
    "unable to connect",
    "proxy error",
    "remote end closed connection",
)

_INTERRUPTED_KEYWORDS = (
    "interrupted",
    "incomplete read",
    "unexpected eof",
    "did not get any data blocks",
    "giving up after",
    "fragment",
    "download cancelled",
    "aborted",
)

_EXTERNAL_TOOL_KEYWORDS = (
    "ffmpeg",
    "ffprobe",
    "postprocessing",
    "postprocessor",
)

_INVALID_URL_KEYWORDS = (
    "is not a valid url",
    "unsupported url",
)

# Prefer H.264 (avc1) + AAC (mp4a/m4a) so merged MP4 files play with sound
# natively on Windows Media Player, Movies & TV, and Qt Multimedia.
VIDEO_MP4_COMPATIBLE_FORMAT = (
    "bestvideo[vcodec^=avc1]+bestaudio[acodec^=mp4a]/"
    "bestvideo[ext=mp4]+bestaudio[ext=m4a]/"
    "best[ext=mp4]/bestvideo+bestaudio/best"
)

_MP4_COMPATIBLE_AUDIO_PREFIXES = ("mp4a", "aac", "mp3")


class _SilentLogger:
    """Captures yt-dlp log messages silently to keep library output clean."""

    def __init__(self) -> None:
        self.warnings: list[str] = []
        self.errors: list[str] = []

    def debug(self, msg: str) -> None:
        return

    def info(self, msg: str) -> None:
        return

    def warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def error(self, msg: str) -> None:
        self.errors.append(msg)


def _find_ffmpeg() -> str | None:
    return shutil.which("ffmpeg")


def _default_ffmpeg_runner(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


class MediaDownloader:
    """Modular video/audio downloader reusable across CLI and future GUI."""

    def __init__(
        self,
        ydl_factory: Callable[[dict[str, Any]], Any] | None = None,
        ffmpeg_checker: Callable[[], str | None] | None = None,
        ffmpeg_runner: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None,
    ) -> None:
        self._ydl_factory = ydl_factory or yt_dlp.YoutubeDL
        self._ffmpeg_checker = ffmpeg_checker or _find_ffmpeg
        self._ffmpeg_runner = ffmpeg_runner or _default_ffmpeg_runner

    def download(
        self,
        request: DownloadRequest,
        progress_callback: ProgressCallback | None = None,
    ) -> DownloadResult:
        """Download video or audio according to the provided request.

        Files are streamed directly to disk by yt-dlp without loading full
        media payloads into memory.
        """
        url = validate_url(request.url)
        output_dir = ensure_output_dir(request.output_dir)
        normalized_audio_format = (
            validate_audio_format(request.audio_format)
            if request.mode == DownloadMode.AUDIO
            else request.audio_format
        )
        normalized_request = DownloadRequest(
            url=url,
            mode=request.mode,
            output_dir=output_dir,
            existing_file_policy=request.existing_file_policy,
            audio_format=normalized_audio_format,
        )

        ffmpeg_path = self._ffmpeg_checker()
        if normalized_request.mode == DownloadMode.AUDIO and not ffmpeg_path:
            err = ExternalToolError(
                "找不到 FFmpeg，無法進行音訊提取與轉檔。請先安裝 FFmpeg 並加入系統 PATH。"
            )
            self._emit_progress(
                progress_callback,
                DownloadProgress(status=DownloadStatus.ERROR, message=str(err)),
            )
            raise err

        self._emit_progress(
            progress_callback,
            DownloadProgress(
                status=DownloadStatus.PREPARING,
                message="正在解析影片資訊...",
            ),
        )

        tracker: dict[str, Any] = {"final_path": None}
        logger = _SilentLogger()
        options = self._build_ydl_options(
            request=normalized_request,
            output_dir=output_dir,
            has_ffmpeg=bool(ffmpeg_path),
            logger=logger,
            progress_callback=progress_callback,
            tracker=tracker,
        )

        try:
            with self._ydl_factory(options) as ydl:
                if normalized_request.existing_file_policy != ExistingFilePolicy.OVERWRITE:
                    probe_info = self._unwrap_entry(
                        ydl.extract_info(url, download=False)
                    )
                    existing_path = self._find_existing_target(
                        ydl=ydl,
                        info=probe_info,
                        request=normalized_request,
                        has_ffmpeg=bool(ffmpeg_path),
                    )
                    if existing_path is not None:
                        if normalized_request.existing_file_policy == ExistingFilePolicy.FAIL:
                            raise FileAlreadyExistsError(str(existing_path))
                        self._emit_progress(
                            progress_callback,
                            DownloadProgress(
                                status=DownloadStatus.SKIPPED,
                                percentage=100.0,
                                filename=str(existing_path),
                                message=f"檔案已存在，略過重新下載：{existing_path.name}",
                            ),
                        )
                        return DownloadResult(
                            url=url,
                            mode=normalized_request.mode,
                            file_path=existing_path,
                            title=str(probe_info.get("title") or existing_path.stem),
                            media_id=probe_info.get("id"),
                            duration=probe_info.get("duration"),
                            file_size=existing_path.stat().st_size,
                            already_exists=True,
                        )

                raw_info = ydl.extract_info(url, download=True)
                info = self._unwrap_entry(raw_info)
                final_path = self._resolve_final_path(
                    ydl=ydl,
                    info=info,
                    request=normalized_request,
                    has_ffmpeg=bool(ffmpeg_path),
                    tracked_path=tracker.get("final_path"),
                )
        except DownloaderError as exc:
            self._emit_progress(
                progress_callback,
                DownloadProgress(status=DownloadStatus.ERROR, message=str(exc)),
            )
            raise
        except KeyboardInterrupt as exc:
            mapped = DownloadInterruptedError("下載已被使用者中斷。")
            self._emit_progress(
                progress_callback,
                DownloadProgress(status=DownloadStatus.ERROR, message=str(mapped)),
            )
            raise mapped from exc
        except Exception as exc:
            mapped = self._classify_exception(exc)
            self._emit_progress(
                progress_callback,
                DownloadProgress(status=DownloadStatus.ERROR, message=str(mapped)),
            )
            raise mapped from exc

        if not final_path.exists() or not final_path.is_file():
            err = MediaDownloadError(
                f"下載程序已結束，但找不到輸出檔案：{final_path}"
            )
            self._emit_progress(
                progress_callback,
                DownloadProgress(status=DownloadStatus.ERROR, message=str(err)),
            )
            raise err

        if (
            normalized_request.mode == DownloadMode.VIDEO
            and ffmpeg_path
            and final_path.suffix.lower() == ".mp4"
        ):
            self._ensure_compatible_mp4_audio(
                final_path=final_path,
                info=info,
                ffmpeg_path=ffmpeg_path,
                progress_callback=progress_callback,
            )

        file_size = final_path.stat().st_size
        self._emit_progress(
            progress_callback,
            DownloadProgress(
                status=DownloadStatus.FINISHED,
                downloaded_bytes=file_size,
                total_bytes=file_size,
                percentage=100.0,
                filename=str(final_path),
                message=f"下載完成：{final_path.name}",
            ),
        )

        return DownloadResult(
            url=url,
            mode=request.mode,
            file_path=final_path,
            title=str(info.get("title") or final_path.stem),
            media_id=info.get("id"),
            duration=info.get("duration"),
            file_size=file_size,
            already_exists=False,
        )

    def _build_ydl_options(
        self,
        *,
        request: DownloadRequest,
        output_dir: Path,
        has_ffmpeg: bool,
        logger: _SilentLogger,
        progress_callback: ProgressCallback | None,
        tracker: dict[str, Any],
    ) -> dict[str, Any]:
        outtmpl = str(output_dir / "%(title)s [%(id)s].%(ext)s")
        overwrites = request.existing_file_policy == ExistingFilePolicy.OVERWRITE

        def _progress_hook(data: dict[str, Any]) -> None:
            status = data.get("status")
            filename = data.get("filename")
            if status == "downloading":
                downloaded = int(data.get("downloaded_bytes") or 0)
                total = data.get("total_bytes") or data.get("total_bytes_estimate")
                total_int = int(total) if total else None
                percentage: float | None = None
                if total_int and total_int > 0:
                    percentage = min(100.0, (downloaded / total_int) * 100.0)
                self._emit_progress(
                    progress_callback,
                    DownloadProgress(
                        status=DownloadStatus.DOWNLOADING,
                        downloaded_bytes=downloaded,
                        total_bytes=total_int,
                        percentage=percentage,
                        speed=data.get("speed"),
                        eta=data.get("eta"),
                        filename=filename,
                        message="下載中...",
                    ),
                )
            elif status == "finished":
                if filename:
                    tracker["final_path"] = Path(filename)
                total = data.get("total_bytes") or data.get("downloaded_bytes")
                total_int = int(total) if total else None
                self._emit_progress(
                    progress_callback,
                    DownloadProgress(
                        status=DownloadStatus.POSTPROCESSING,
                        downloaded_bytes=total_int or 0,
                        total_bytes=total_int,
                        percentage=100.0,
                        filename=filename,
                        message="檔案下載完成，正在進行後處理...",
                    ),
                )
            elif status == "error":
                self._emit_progress(
                    progress_callback,
                    DownloadProgress(
                        status=DownloadStatus.ERROR,
                        filename=filename,
                        message="下載過程中發生錯誤。",
                    ),
                )

        def _postprocessor_hook(data: dict[str, Any]) -> None:
            status = data.get("status")
            info_dict = data.get("info_dict") or {}
            filepath = info_dict.get("filepath")
            if status in {"started", "processing"}:
                self._emit_progress(
                    progress_callback,
                    DownloadProgress(
                        status=DownloadStatus.POSTPROCESSING,
                        percentage=100.0,
                        filename=filepath,
                        message=f"後處理中：{data.get('postprocessor') or 'FFmpeg'}...",
                    ),
                )
            elif status == "finished" and filepath:
                tracker["final_path"] = Path(filepath)

        options: dict[str, Any] = {
            "outtmpl": outtmpl,
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "windowsfilenames": True,
            "overwrites": overwrites,
            "retries": 3,
            "fragment_retries": 3,
            "socket_timeout": 30,
            "logger": logger,
            "progress_hooks": [_progress_hook],
            "postprocessor_hooks": [_postprocessor_hook],
        }

        if request.mode == DownloadMode.AUDIO:
            options["format"] = "bestaudio/best"
            options["postprocessors"] = [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": request.audio_format,
                }
            ]
        else:
            if has_ffmpeg:
                options["format"] = VIDEO_MP4_COMPATIBLE_FORMAT
                options["merge_output_format"] = "mp4"
            else:
                options["format"] = "best[ext=mp4]/best"

        return options

    def _ensure_compatible_mp4_audio(
        self,
        *,
        final_path: Path,
        info: dict[str, Any],
        ffmpeg_path: str,
        progress_callback: ProgressCallback | None,
    ) -> None:
        """Transcode non-AAC audio (e.g. Opus/Vorbis) inside MP4 to AAC for Windows playback."""
        acodec = self._extract_audio_codec(info)
        if not acodec:
            return
        if acodec.startswith(_MP4_COMPATIBLE_AUDIO_PREFIXES):
            return

        self._emit_progress(
            progress_callback,
            DownloadProgress(
                status=DownloadStatus.POSTPROCESSING,
                percentage=100.0,
                filename=str(final_path),
                message=f"偵測到 MP4 音訊編碼為 {acodec}，正在轉為相容性較佳的 AAC 音訊...",
            ),
        )

        temp_output = final_path.with_name(f"{final_path.stem}.aac_tmp{final_path.suffix}")
        cmd = [
            ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(final_path),
            "-map",
            "0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(temp_output),
        ]
        try:
            proc = self._ffmpeg_runner(cmd)
        except OSError as exc:
            temp_output.unlink(missing_ok=True)
            err = ExternalToolError(f"執行 FFmpeg 轉碼 MP4 音訊失敗：{exc}")
            self._emit_progress(
                progress_callback,
                DownloadProgress(status=DownloadStatus.ERROR, message=str(err)),
            )
            raise err from exc

        if proc.returncode != 0 or not temp_output.exists() or temp_output.stat().st_size == 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            temp_output.unlink(missing_ok=True)
            err = ExternalToolError(
                f"FFmpeg 轉碼 MP4 音訊為 AAC 失敗：{detail or f'exit code {proc.returncode}'}"
            )
            self._emit_progress(
                progress_callback,
                DownloadProgress(status=DownloadStatus.ERROR, message=str(err)),
            )
            raise err

        try:
            temp_output.replace(final_path)
        except OSError as exc:
            temp_output.unlink(missing_ok=True)
            err = MediaDownloadError(f"更新轉碼後的 MP4 檔案失敗：{exc}")
            self._emit_progress(
                progress_callback,
                DownloadProgress(status=DownloadStatus.ERROR, message=str(err)),
            )
            raise err from exc

    @staticmethod
    def _extract_audio_codec(info: dict[str, Any]) -> str | None:
        raw_codecs: list[Any] = [info.get("acodec")]
        requested_downloads = info.get("requested_downloads") or []
        for item in requested_downloads:
            if isinstance(item, dict):
                raw_codecs.append(item.get("acodec"))
                for fmt in item.get("requested_formats") or []:
                    if isinstance(fmt, dict):
                        raw_codecs.append(fmt.get("acodec"))
        for fmt in info.get("requested_formats") or []:
            if isinstance(fmt, dict):
                raw_codecs.append(fmt.get("acodec"))

        for codec in raw_codecs:
            if isinstance(codec, str):
                cleaned = codec.strip().lower()
                if cleaned and cleaned != "none":
                    return cleaned
        return None

    @staticmethod
    def _unwrap_entry(info: Any) -> dict[str, Any]:
        if not isinstance(info, dict):
            raise MediaDownloadError("無法取得有效的影片中繼資料。")
        if "entries" in info and info["entries"]:
            entries = [e for e in info["entries"] if isinstance(e, dict)]
            if entries:
                return entries[0]
        return info

    @staticmethod
    def _expected_candidates(
        *,
        ydl: Any,
        info: dict[str, Any],
        request: DownloadRequest,
        has_ffmpeg: bool,
    ) -> list[Path]:
        candidates: list[Path] = []
        prepared = ydl.prepare_filename(info)
        if prepared:
            prepared_path = Path(prepared)
            if request.mode == DownloadMode.AUDIO:
                candidates.append(prepared_path.with_suffix(f".{request.audio_format}"))
            elif has_ffmpeg:
                candidates.append(prepared_path.with_suffix(".mp4"))
                candidates.append(prepared_path)
            else:
                candidates.append(prepared_path)
        return candidates

    def _find_existing_target(
        self,
        *,
        ydl: Any,
        info: dict[str, Any],
        request: DownloadRequest,
        has_ffmpeg: bool,
    ) -> Path | None:
        for candidate in self._expected_candidates(
            ydl=ydl, info=info, request=request, has_ffmpeg=has_ffmpeg
        ):
            if candidate.exists() and candidate.is_file():
                return candidate
        return None

    def _resolve_final_path(
        self,
        *,
        ydl: Any,
        info: dict[str, Any],
        request: DownloadRequest,
        has_ffmpeg: bool,
        tracked_path: Path | None,
    ) -> Path:
        requested_downloads = info.get("requested_downloads") or []
        for item in requested_downloads:
            if isinstance(item, dict) and item.get("filepath"):
                candidate = Path(item["filepath"])
                if candidate.exists():
                    return candidate

        if tracked_path is not None and tracked_path.exists():
            return tracked_path

        candidates = self._expected_candidates(
            ydl=ydl, info=info, request=request, has_ffmpeg=has_ffmpeg
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate

        if requested_downloads and isinstance(requested_downloads[0], dict):
            fp = requested_downloads[0].get("filepath")
            if fp:
                return Path(fp)

        if tracked_path is not None:
            return tracked_path

        if candidates:
            return candidates[0]

        raise MediaDownloadError("無法判斷下載後的檔案路徑。")

    @staticmethod
    def _classify_exception(exc: Exception) -> DownloaderError:
        if isinstance(exc, DownloaderError):
            return exc

        if isinstance(exc, PostProcessingError):
            return ExternalToolError(f"媒體後處理失敗（請檢查 FFmpeg）：{exc}")

        if isinstance(exc, (ConnectionError, TimeoutError, URLError)):
            return NetworkDownloadError(f"網路連線失敗，請檢查網路狀態：{exc}")

        msg = str(exc)
        lower = msg.lower()

        if isinstance(exc, FileNotFoundError) and any(
            kw in lower for kw in _EXTERNAL_TOOL_KEYWORDS
        ):
            return ExternalToolError(f"找不到外部工具（FFmpeg）：{msg}")

        if any(kw in lower for kw in _INVALID_URL_KEYWORDS):
            return InvalidURLError(f"無法識別或目前不支援此影片網址：{msg}")

        if any(kw in lower for kw in _NETWORK_KEYWORDS):
            return NetworkDownloadError(f"網路連線失敗或連線逾時：{msg}")

        if any(kw in lower for kw in _INTERRUPTED_KEYWORDS):
            return DownloadInterruptedError(f"下載中斷或傳輸未完成：{msg}")

        if any(kw in lower for kw in _EXTERNAL_TOOL_KEYWORDS):
            return ExternalToolError(f"外部工具執行失敗（FFmpeg）：{msg}")

        if isinstance(exc, (DownloadError, ExtractorError)):
            return MediaDownloadError(f"影片下載或解析失敗：{msg}")

        return MediaDownloadError(f"下載時發生未預期的錯誤：{msg}")

    @staticmethod
    def _emit_progress(
        callback: ProgressCallback | None,
        progress: DownloadProgress,
    ) -> None:
        if callback is None:
            return
        try:
            callback(progress)
        except Exception:
            # Progress display issues must never crash or hide the core download flow.
            return
