"""Public API for the SubtitleUI media downloader module."""

from .errors import (
    DownloaderError,
    DownloadInterruptedError,
    ExternalToolError,
    FileAlreadyExistsError,
    InvalidOutputDirectoryError,
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
from .service import MediaDownloader
from .validators import ensure_output_dir, validate_url

__all__ = [
    "DownloadInterruptedError",
    "DownloadMode",
    "DownloadProgress",
    "DownloadRequest",
    "DownloadResult",
    "DownloadStatus",
    "DownloaderError",
    "ExistingFilePolicy",
    "ExternalToolError",
    "FileAlreadyExistsError",
    "InvalidOutputDirectoryError",
    "InvalidURLError",
    "MediaDownloadError",
    "MediaDownloader",
    "NetworkDownloadError",
    "ProgressCallback",
    "ensure_output_dir",
    "validate_url",
]
