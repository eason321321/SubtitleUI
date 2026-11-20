"""Data models for media download requests, progress updates, and results."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable


class DownloadMode(str, Enum):
    """Supported download modes."""

    VIDEO = "video"
    AUDIO = "audio"


class ExistingFilePolicy(str, Enum):
    """How to handle a target file that already exists on disk."""

    SKIP = "skip"
    OVERWRITE = "overwrite"
    FAIL = "fail"


class DownloadStatus(str, Enum):
    """Lifecycle states emitted through download progress callbacks."""

    PREPARING = "preparing"
    DOWNLOADING = "downloading"
    POSTPROCESSING = "postprocessing"
    FINISHED = "finished"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass(frozen=True)
class DownloadRequest:
    """Input specification for a single media download."""

    url: str
    mode: DownloadMode = DownloadMode.VIDEO
    output_dir: Path = Path("downloads")
    existing_file_policy: ExistingFilePolicy = ExistingFilePolicy.SKIP
    audio_format: str = "m4a"


@dataclass(frozen=True)
class DownloadProgress:
    """Structured progress snapshot emitted during download."""

    status: DownloadStatus
    downloaded_bytes: int = 0
    total_bytes: int | None = None
    percentage: float | None = None
    speed: float | None = None
    eta: int | None = None
    filename: str | None = None
    message: str = ""


@dataclass(frozen=True)
class DownloadResult:
    """Final outcome of a completed or skipped media download."""

    url: str
    mode: DownloadMode
    file_path: Path
    title: str
    media_id: str | None = None
    duration: float | None = None
    file_size: int | None = None
    already_exists: bool = False


ProgressCallback = Callable[[DownloadProgress], None]
