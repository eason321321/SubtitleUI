"""Validation helpers for URLs and download output directories."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from .errors import (
    InvalidOutputDirectoryError,
    InvalidURLError,
    MediaDownloadError,
)

_ALLOWED_SCHEMES = {"http", "https"}


def validate_url(url: str) -> str:
    """Validate and normalize a media URL.

    Args:
        url: Raw URL input from the caller.

    Returns:
        Normalized URL string with surrounding whitespace stripped.

    Raises:
        InvalidURLError: If the URL is empty, malformed, or uses an unsupported scheme.
    """
    if not isinstance(url, str):
        raise InvalidURLError("影片網址必須為文字格式。")

    cleaned = url.strip()
    if not cleaned:
        raise InvalidURLError("影片網址不可為空白。")

    if any(ch.isspace() for ch in cleaned):
        raise InvalidURLError(f"影片網址格式不正確（包含空白字元）：{cleaned}")

    parsed = urlparse(cleaned)
    if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
        raise InvalidURLError(
            f"不支援的網址協定「{parsed.scheme or '無'}」，僅支援 http 或 https。"
        )

    hostname = parsed.hostname
    if not hostname or hostname.startswith(".") or hostname.endswith("."):
        raise InvalidURLError(f"影片網址缺少有效的主機名稱：{cleaned}")

    if "." not in hostname and hostname != "localhost":
        raise InvalidURLError(f"影片網址的主機名稱格式不正確：{cleaned}")

    return cleaned


def ensure_output_dir(output_dir: str | Path) -> Path:
    """Ensure the download output directory exists and is usable.

    Args:
        output_dir: Target directory path.

    Returns:
        Resolved Path pointing to an existing directory.

    Raises:
        InvalidOutputDirectoryError: If the path is invalid, points to a file,
            or cannot be created.
    """
    if not output_dir or (isinstance(output_dir, str) and not output_dir.strip()):
        raise InvalidOutputDirectoryError("輸出資料夾路徑不可為空白。")

    try:
        target = Path(output_dir).expanduser().resolve()
    except OSError as exc:
        raise InvalidOutputDirectoryError(f"無法解析輸出資料夾路徑：{output_dir}") from exc

    if target.exists() and not target.is_dir():
        raise InvalidOutputDirectoryError(
            f"輸出路徑已存在且不是資料夾：{target}"
        )

    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise InvalidOutputDirectoryError(
            f"無法建立輸出資料夾「{target}」：{exc}"
        ) from exc

    return target


def validate_audio_format(audio_format: str) -> str:
    """Validate and normalize an audio output codec/extension name."""
    if not isinstance(audio_format, str):
        raise MediaDownloadError("音訊格式必須為文字。")
    cleaned = audio_format.strip().lstrip(".").lower()
    if not cleaned or not cleaned.isalnum():
        raise MediaDownloadError(
            f"不支援的音訊格式「{audio_format}」，請使用英數字格式名稱（例如 m4a、mp3、wav）。"
        )
    return cleaned

