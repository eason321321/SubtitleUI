"""Unit tests for the SubtitleUI Phase 1 CLI interface."""

from __future__ import annotations

import io
import sys
import tempfile
import unittest
from pathlib import Path

from yt_dlp.utils import DownloadError

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from subtitleui.cli import main  # noqa: E402
from subtitleui.downloader import MediaDownloader  # noqa: E402
from tests.test_downloader import FakeYoutubeDL  # noqa: E402


class TestCLI(unittest.TestCase):
    """Tests for CLI argument parsing, interactive input, and error safety."""

    def test_cli_downloads_video_successfully(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            stderr = io.StringIO()
            downloader = MediaDownloader(
                ydl_factory=lambda opts: FakeYoutubeDL(opts),
                ffmpeg_checker=lambda: "ffmpeg",
            )

            exit_code = main(
                [
                    "https://www.youtube.com/watch?v=abc123XYZ",
                    "--mode",
                    "video",
                    "--output",
                    tmp,
                ],
                downloader=downloader,
                stdout=stdout,
                stderr=stderr,
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(stderr.getvalue(), "")
            output = stdout.getvalue()
            self.assertIn("下載成功", output)
            self.assertIn("한국어 강의 샘플", output)
            self.assertIn("[下載中]", output)

    def test_cli_interactive_audio_download(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            stderr = io.StringIO()
            responses = iter(
                ["https://www.youtube.com/watch?v=abc123XYZ", "audio", tmp]
            )
            downloader = MediaDownloader(
                ydl_factory=lambda opts: FakeYoutubeDL(opts, ext="webm"),
                ffmpeg_checker=lambda: "ffmpeg",
            )

            exit_code = main(
                [],
                downloader=downloader,
                input_func=lambda: next(responses),
                stdout=stdout,
                stderr=stderr,
            )

            self.assertEqual(exit_code, 0)
            self.assertIn("模式：audio", stdout.getvalue())
            self.assertIn(".m4a", stdout.getvalue())

    def test_cli_handles_invalid_url_without_crashing(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        downloader = MediaDownloader(
            ydl_factory=lambda opts: FakeYoutubeDL(opts),
            ffmpeg_checker=lambda: "ffmpeg",
        )

        exit_code = main(
            ["not-a-valid-url"],
            downloader=downloader,
            stdout=stdout,
            stderr=stderr,
        )

        self.assertEqual(exit_code, 1)
        self.assertIn("下載失敗", stderr.getvalue())

    def test_cli_handles_network_failure_gracefully(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            stderr = io.StringIO()
            downloader = MediaDownloader(
                ydl_factory=lambda opts: FakeYoutubeDL(
                    opts,
                    error_on_extract=DownloadError("ERROR: timed out"),
                ),
                ffmpeg_checker=lambda: "ffmpeg",
            )

            exit_code = main(
                ["https://www.youtube.com/watch?v=netfail", "-o", tmp],
                downloader=downloader,
                stdout=stdout,
                stderr=stderr,
            )

            self.assertEqual(exit_code, 1)
            self.assertIn("網路連線失敗", stderr.getvalue())

    def test_cli_handles_keyboard_interrupt_gracefully(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            stderr = io.StringIO()
            downloader = MediaDownloader(
                ydl_factory=lambda opts: FakeYoutubeDL(
                    opts,
                    error_on_download=KeyboardInterrupt(),
                ),
                ffmpeg_checker=lambda: "ffmpeg",
            )

            exit_code = main(
                ["https://www.youtube.com/watch?v=kb", "-o", tmp],
                downloader=downloader,
                stdout=stdout,
                stderr=stderr,
            )

            self.assertEqual(exit_code, 130)
            self.assertIn("下載中斷", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
