"""Custom exception types for the SubtitleUI downloader module."""


class DownloaderError(Exception):
    """Base exception for all media download errors."""


class InvalidURLError(DownloaderError):
    """Raised when the provided media URL is empty, malformed, or unsupported."""


class InvalidOutputDirectoryError(DownloaderError):
    """Raised when the output directory cannot be created or is not a directory."""


class FileAlreadyExistsError(DownloaderError):
    """Raised when the target output file already exists and policy is FAIL."""

    def __init__(self, file_path: str, message: str | None = None) -> None:
        self.file_path = file_path
        super().__init__(message or f"目標檔案已存在：{file_path}")


class NetworkDownloadError(DownloaderError):
    """Raised when a network connectivity, DNS, or timeout error occurs."""


class DownloadInterruptedError(DownloaderError):
    """Raised when a download is interrupted by the user or stops mid-transfer."""


class ExternalToolError(DownloaderError):
    """Raised when an external tool (such as FFmpeg) is missing or fails."""


class MediaDownloadError(DownloaderError):
    """Raised for general media extraction or download failures."""
