"""FFmpeg/FFprobe audio probing and accurate time-range WAV extraction."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
import wave
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from .errors import AudioExtractionError, AudioInputError
from .models import CancellationToken, TimeRange

ASR_SAMPLE_RATE = 16000
ASR_CHANNELS = 1
ASR_AUDIO_CODEC = "pcm_s16le"

CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class ExtractedAudio:
    """Verified temporary WAV audio prepared for ASR inference."""

    wav_path: Path
    duration: float
    sample_rate: int = ASR_SAMPLE_RATE
    channels: int = ASR_CHANNELS


def _find_ffmpeg() -> str | None:
    return shutil.which("ffmpeg")


def _find_ffprobe() -> str | None:
    return shutil.which("ffprobe")


def _default_runner(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _read_wav_info(wav_path: Path) -> tuple[float, int, int]:
    """Read exact duration, sample rate, and channel count from a PCM WAV file."""
    try:
        with wave.open(str(wav_path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
    except (OSError, wave.Error) as exc:
        raise AudioExtractionError(
            f"無法讀取擷取後的 WAV 音訊檔「{wav_path}」：{exc}"
        ) from exc

    if sample_rate <= 0:
        raise AudioExtractionError(f"擷取後的 WAV 取樣率無效：{sample_rate}")

    duration = frame_count / float(sample_rate)
    return round(duration, 3), sample_rate, channels


class AudioExtractor:
    """Probes media duration and extracts 16 kHz mono PCM WAV slices via FFmpeg."""

    def __init__(
        self,
        *,
        ffmpeg_finder: Callable[[], str | None] | None = None,
        ffprobe_finder: Callable[[], str | None] | None = None,
        runner: CommandRunner | None = None,
        wav_inspector: Callable[[Path], tuple[float, int, int]] | None = None,
    ) -> None:
        self._ffmpeg_finder = ffmpeg_finder or _find_ffmpeg
        self._ffprobe_finder = ffprobe_finder or _find_ffprobe
        self._runner = runner or _default_runner
        self._wav_inspector = wav_inspector or _read_wav_info

    def probe_duration(self, media_path: str | Path) -> float:
        """Verify media file existence, audio stream presence, and total duration."""
        source = self._validate_media_path(media_path)

        ffprobe = self._ffprobe_finder()
        if not ffprobe:
            raise AudioExtractionError(
                "找不到 FFprobe，無法讀取媒體長度。請安裝 FFmpeg 並加入系統 PATH。"
            )

        cmd = [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type,duration",
            "-of",
            "json",
            str(source),
        ]
        try:
            proc = self._runner(cmd)
        except OSError as exc:
            raise AudioExtractionError(f"執行 FFprobe 失敗：{exc}") from exc

        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            raise AudioInputError(
                f"無法解析媒體檔案「{source.name}」，檔案可能已損毀或不支援：{detail or 'ffprobe 回傳非零狀態碼'}"
            )

        try:
            payload: dict[str, Any] = json.loads(proc.stdout or "{}")
        except json.JSONDecodeError as exc:
            raise AudioExtractionError(f"無法解析 FFprobe 輸出：{exc}") from exc

        streams = payload.get("streams") or []
        audio_streams = [
            stream
            for stream in streams
            if isinstance(stream, dict) and stream.get("codec_type") == "audio"
        ]
        if not audio_streams:
            raise AudioInputError(f"檔案中沒有可辨識的音訊軌道：{source}")

        duration = self._extract_duration_from_probe(payload, audio_streams)
        if duration is None or not math.isfinite(duration) or duration <= 0:
            raise AudioInputError(f"無法取得有效的音訊長度或音訊長度為 0：{source}")

        return round(duration, 3)

    def extract_wav(
        self,
        media_path: str | Path,
        output_wav: Path,
        *,
        time_range: TimeRange,
        is_custom_range: bool,
        cancel_token: CancellationToken | None = None,
    ) -> ExtractedAudio:
        """Decode and extract audio into a 16 kHz mono PCM WAV file on disk.

        Uses full audio decoding (never stream copy) and accurate seeking so
        time boundaries align cleanly with the requested range.
        """
        if cancel_token is not None:
            cancel_token.raise_if_cancelled()

        source = self._validate_media_path(media_path)
        ffmpeg = self._ffmpeg_finder()
        if not ffmpeg:
            raise AudioExtractionError(
                "找不到 FFmpeg，無法擷取音訊。請先安裝 FFmpeg 並加入系統 PATH。"
            )

        output_wav.parent.mkdir(parents=True, exist_ok=True)

        # Place -ss after -i so FFmpeg decodes to the exact requested timestamp
        # instead of snapping to keyframes or using inaccurate stream copy.
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
        ]
        if is_custom_range:
            cmd.extend(
                [
                    "-ss",
                    f"{time_range.start:.3f}",
                    "-t",
                    f"{time_range.duration:.3f}",
                ]
            )
        cmd.extend(
            [
                "-vn",
                "-sn",
                "-dn",
                "-ac",
                str(ASR_CHANNELS),
                "-ar",
                str(ASR_SAMPLE_RATE),
                "-c:a",
                ASR_AUDIO_CODEC,
                str(output_wav),
            ]
        )

        try:
            proc = self._runner(cmd)
        except OSError as exc:
            raise AudioExtractionError(f"執行 FFmpeg 失敗：{exc}") from exc

        if cancel_token is not None:
            cancel_token.raise_if_cancelled()

        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            raise AudioExtractionError(
                f"FFmpeg 擷取音訊失敗：{detail or f'exit code {proc.returncode}'}"
            )

        if not output_wav.exists() or output_wav.stat().st_size == 0:
            raise AudioExtractionError(f"FFmpeg 執行完成但未產生有效的音訊檔案：{output_wav}")

        actual_duration, sample_rate, channels = self._wav_inspector(output_wav)
        if actual_duration <= 0:
            raise AudioExtractionError("擷取後的音訊長度為 0 秒，無法進行語音辨識。")

        expected_duration = time_range.duration
        tolerance = max(0.5, expected_duration * 0.05)
        if abs(actual_duration - expected_duration) > tolerance:
            raise AudioExtractionError(
                f"擷取後的音訊長度（{actual_duration:.2f}s）與預期區間（{expected_duration:.2f}s）落差過大。"
            )

        return ExtractedAudio(
            wav_path=output_wav,
            duration=actual_duration,
            sample_rate=sample_rate,
            channels=channels,
        )

    @contextmanager
    def extract_to_temp(
        self,
        media_path: str | Path,
        *,
        time_range: TimeRange,
        is_custom_range: bool,
        cancel_token: CancellationToken | None = None,
    ) -> Iterator[ExtractedAudio]:
        """Extract audio into an auto-cleaned temporary directory."""
        with tempfile.TemporaryDirectory(prefix="subtitleui_asr_") as tmp_dir:
            wav_path = Path(tmp_dir) / "segment_16k_mono.wav"
            extracted = self.extract_wav(
                media_path,
                wav_path,
                time_range=time_range,
                is_custom_range=is_custom_range,
                cancel_token=cancel_token,
            )
            yield extracted

    @staticmethod
    def _validate_media_path(media_path: str | Path) -> Path:
        if not media_path or (isinstance(media_path, str) and not media_path.strip()):
            raise AudioInputError("音訊或影片檔案路徑不可為空白。")
        path = Path(media_path).expanduser().resolve()
        if not path.exists():
            raise AudioInputError(f"找不到音訊或影片檔案：{path}")
        if not path.is_file():
            raise AudioInputError(f"輸入路徑不是有效的檔案：{path}")
        return path

    @staticmethod
    def _extract_duration_from_probe(
        payload: dict[str, Any],
        audio_streams: list[dict[str, Any]],
    ) -> float | None:
        candidates: list[Any] = []
        for stream in audio_streams:
            if stream.get("duration") is not None:
                candidates.append(stream["duration"])
        fmt = payload.get("format")
        if isinstance(fmt, dict) and fmt.get("duration") is not None:
            candidates.append(fmt["duration"])

        for raw in candidates:
            try:
                value = float(raw)
                if math.isfinite(value) and value > 0:
                    return value
            except (TypeError, ValueError):
                continue
        return None
