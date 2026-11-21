"""Unit tests for the SubtitleUI Phase 1 downloader module."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from urllib.error import URLError

from yt_dlp.utils import DownloadError, PostProcessingError

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from subtitleui.downloader import (  # noqa: E402
    DownloadInterruptedError,
    DownloadMode,
    DownloadProgress,
    DownloadRequest,
    DownloadStatus,
    ExistingFilePolicy,
    ExternalToolError,
    FileAlreadyExistsError,
    InvalidOutputDirectoryError,
    InvalidURLError,
    MediaDownloadError,
    MediaDownloader,
    NetworkDownloadError,
    ensure_output_dir,
    validate_url,
)


from subtitleui.downloader.service import VIDEO_MP4_COMPATIBLE_FORMAT  # noqa: E402
from subtitleui.paths import DOWNLOAD_DIR  # noqa: E402


class FakeYoutubeDL:
    """Deterministic yt-dlp fake that writes tiny placeholder files on disk."""

    def __init__(
        self,
        options: dict[str, Any],
        *,
        title: str = "한국어 강의 샘플",
        media_id: str = "abc123XYZ",
        ext: str = "mp4",
        duration: float = 125.0,
        payload: bytes = b"fake-media-stream-bytes",
        error_on_extract: Exception | None = None,
        error_on_download: Exception | None = None,
        skip_file_creation: bool = False,
        record_calls: list[dict[str, Any]] | None = None,
        acodec: str | None = "mp4a.40.2",
    ) -> None:
        self.options = options
        self.title = title
        self.media_id = media_id
        self.ext = ext
        self.duration = duration
        self.payload = payload
        self.error_on_extract = error_on_extract
        self.error_on_download = error_on_download
        self.skip_file_creation = skip_file_creation
        self.record_calls = record_calls if record_calls is not None else []
        self.acodec = acodec

    def __enter__(self) -> FakeYoutubeDL:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def prepare_filename(self, info: dict[str, Any]) -> str:
        outtmpl: str = self.options["outtmpl"]
        return outtmpl % {
            "title": info.get("title", self.title),
            "id": info.get("id", self.media_id),
            "ext": info.get("ext", self.ext),
        }

    def extract_info(self, url: str, download: bool = True) -> dict[str, Any]:
        self.record_calls.append(
            {"url": url, "download": download, "options": dict(self.options)}
        )

        if self.error_on_extract is not None:
            raise self.error_on_extract

        info: dict[str, Any] = {
            "id": self.media_id,
            "title": self.title,
            "ext": self.ext,
            "duration": self.duration,
            "webpage_url": url,
            "acodec": self.acodec,
        }

        if not download:
            return info

        if self.error_on_download is not None:
            raise self.error_on_download

        prepared = Path(self.prepare_filename(info))
        final_path = prepared
        postprocessors = self.options.get("postprocessors") or []
        for pp in postprocessors:
            if pp.get("key") == "FFmpegExtractAudio":
                codec = pp.get("preferredcodec", "m4a")
                final_path = prepared.with_suffix(f".{codec}")
        if self.options.get("merge_output_format"):
            final_path = prepared.with_suffix(f".{self.options['merge_output_format']}")

        for hook in self.options.get("progress_hooks", []):
            hook(
                {
                    "status": "downloading",
                    "downloaded_bytes": len(self.payload) // 2,
                    "total_bytes": len(self.payload),
                    "speed": 1024.0,
                    "eta": 2,
                    "filename": str(prepared),
                }
            )
            hook(
                {
                    "status": "finished",
                    "downloaded_bytes": len(self.payload),
                    "total_bytes": len(self.payload),
                    "filename": str(prepared),
                }
            )

        if not self.skip_file_creation:
            final_path.parent.mkdir(parents=True, exist_ok=True)
            final_path.write_bytes(self.payload)

        for pp_hook in self.options.get("postprocessor_hooks", []):
            pp_hook(
                {
                    "status": "started",
                    "postprocessor": "FFmpeg",
                    "info_dict": {"filepath": str(final_path)},
                }
            )
            pp_hook(
                {
                    "status": "finished",
                    "postprocessor": "FFmpeg",
                    "info_dict": {"filepath": str(final_path)},
                }
            )

        info["requested_downloads"] = [{"filepath": str(final_path)}]
        return info


class TestValidators(unittest.TestCase):
    """Tests for URL and output directory validators."""

    def test_valid_urls(self) -> None:
        self.assertEqual(
            validate_url("  https://www.youtube.com/watch?v=dQw4w9WgXcQ  "),
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        )
        self.assertEqual(
            validate_url("http://youtu.be/abc123"),
            "http://youtu.be/abc123",
        )

    def test_invalid_urls_raise_invalid_url_error(self) -> None:
        bad_urls = [
            "",
            "   ",
            "not-a-url",
            "ftp://example.com/video.mp4",
            "file:///C:/video.mp4",
            "https://",
            "https://invalidhost",
            "https://example.com/has space",
            "https://.example.com/video",
        ]
        for raw in bad_urls:
            with self.subTest(url=raw):
                with self.assertRaises(InvalidURLError):
                    validate_url(raw)

    def test_ensure_output_dir_creates_missing_nested_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            nested = Path(tmp) / "level1" / "level2" / "downloads"
            self.assertFalse(nested.exists())
            created = ensure_output_dir(nested)
            self.assertTrue(created.exists())
            self.assertTrue(created.is_dir())

    def test_ensure_output_dir_fails_when_path_is_a_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            file_path = Path(tmp) / "not_a_dir.txt"
            file_path.write_text("hello", encoding="utf-8")
            with self.assertRaises(InvalidOutputDirectoryError):
                ensure_output_dir(file_path)


class TestMediaDownloader(unittest.TestCase):
    """Tests covering all Phase 1 download modes, policies, and failure cases."""

    def test_valid_video_download_with_progress_and_missing_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "new_downloads"
            calls: list[dict[str, Any]] = []
            events: list[DownloadProgress] = []

            downloader = MediaDownloader(
                ydl_factory=lambda opts: FakeYoutubeDL(
                    opts,
                    title="한국어 듣기 연습",
                    media_id="kor001",
                    ext="webm",
                    record_calls=calls,
                ),
                ffmpeg_checker=lambda: "C:/ffmpeg/bin/ffmpeg.exe",
            )

            result = downloader.download(
                DownloadRequest(
                    url="https://www.youtube.com/watch?v=kor001",
                    mode=DownloadMode.VIDEO,
                    output_dir=output_dir,
                ),
                progress_callback=events.append,
            )

            self.assertFalse(result.already_exists)
            self.assertEqual(result.mode, DownloadMode.VIDEO)
            self.assertEqual(result.title, "한국어 듣기 연습")
            self.assertEqual(result.media_id, "kor001")
            self.assertTrue(result.file_path.exists())
            self.assertEqual(result.file_path.suffix, ".mp4")
            self.assertEqual(result.file_path.parent, output_dir.resolve())
            self.assertGreater(result.file_size or 0, 0)

            statuses = [event.status for event in events]
            self.assertIn(DownloadStatus.PREPARING, statuses)
            self.assertIn(DownloadStatus.DOWNLOADING, statuses)
            self.assertIn(DownloadStatus.POSTPROCESSING, statuses)
            self.assertEqual(statuses[-1], DownloadStatus.FINISHED)

            # Verify yt-dlp video options with FFmpeg available prefer Windows-compatible MP4 codecs
            dl_opts = calls[-1]["options"]
            self.assertEqual(dl_opts["format"], VIDEO_MP4_COMPATIBLE_FORMAT)
            self.assertIn("vcodec^=avc1", dl_opts["format"])
            self.assertIn("acodec^=mp4a", dl_opts["format"])
            self.assertEqual(dl_opts["merge_output_format"], "mp4")
            self.assertTrue(dl_opts["noplaylist"])
            self.assertTrue(dl_opts["windowsfilenames"])

    def test_video_mp4_skips_transcode_when_audio_is_already_aac(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ffmpeg_cmds: list[list[str]] = []

            def fake_ffmpeg(cmd: list[str]) -> subprocess.CompletedProcess[str]:
                ffmpeg_cmds.append(cmd)
                return subprocess.CompletedProcess(cmd, 0, "", "")

            downloader = MediaDownloader(
                ydl_factory=lambda opts: FakeYoutubeDL(opts, acodec="mp4a.40.2"),
                ffmpeg_checker=lambda: "ffmpeg",
                ffmpeg_runner=fake_ffmpeg,
            )
            result = downloader.download(
                DownloadRequest(
                    url="https://www.youtube.com/watch?v=aac001",
                    mode=DownloadMode.VIDEO,
                    output_dir=Path(tmp),
                )
            )
            self.assertTrue(result.file_path.exists())
            self.assertEqual(ffmpeg_cmds, [])

    def test_video_mp4_transcodes_opus_audio_to_aac_for_windows_compatibility(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ffmpeg_cmds: list[list[str]] = []

            def fake_ffmpeg(cmd: list[str]) -> subprocess.CompletedProcess[str]:
                ffmpeg_cmds.append(cmd)
                Path(cmd[-1]).write_bytes(b"mp4-with-aac-audio")
                return subprocess.CompletedProcess(cmd, 0, "", "")

            downloader = MediaDownloader(
                ydl_factory=lambda opts: FakeYoutubeDL(opts, acodec="opus"),
                ffmpeg_checker=lambda: "ffmpeg",
                ffmpeg_runner=fake_ffmpeg,
            )
            result = downloader.download(
                DownloadRequest(
                    url="https://www.youtube.com/watch?v=opus001",
                    mode=DownloadMode.VIDEO,
                    output_dir=Path(tmp),
                )
            )
            self.assertEqual(len(ffmpeg_cmds), 1)
            self.assertIn("-c:v", ffmpeg_cmds[0])
            self.assertIn("copy", ffmpeg_cmds[0])
            self.assertIn("-c:a", ffmpeg_cmds[0])
            self.assertIn("aac", ffmpeg_cmds[0])
            self.assertEqual(result.file_path.read_bytes(), b"mp4-with-aac-audio")
            self.assertEqual(result.file_size, len(b"mp4-with-aac-audio"))

    def test_default_output_dir_and_temp_dir_isolation_for_mp4_transcode(self) -> None:
        req = DownloadRequest(url="https://www.youtube.com/watch?v=def001")
        self.assertEqual(req.output_dir, DOWNLOAD_DIR)

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "downloads"
            temp_dir = Path(tmp) / "temp"
            ffmpeg_temp_targets: list[Path] = []

            def fake_ffmpeg(cmd: list[str]) -> subprocess.CompletedProcess[str]:
                target = Path(cmd[-1])
                ffmpeg_temp_targets.append(target)
                target.write_bytes(b"transcoded-aac-mp4")
                return subprocess.CompletedProcess(cmd, 0, "", "")

            downloader = MediaDownloader(
                ydl_factory=lambda opts: FakeYoutubeDL(opts, acodec="opus"),
                ffmpeg_checker=lambda: "ffmpeg",
                ffmpeg_runner=fake_ffmpeg,
                temp_dir=temp_dir,
            )
            result = downloader.download(
                DownloadRequest(
                    url="https://www.youtube.com/watch?v=opus002",
                    mode=DownloadMode.VIDEO,
                    output_dir=out_dir,
                )
            )
            self.assertEqual(len(ffmpeg_temp_targets), 1)
            # Temp file was created inside temp_dir, not downloads/
            self.assertEqual(ffmpeg_temp_targets[0].parent.parent, temp_dir.resolve())
            self.assertFalse(ffmpeg_temp_targets[0].exists())
            self.assertFalse(ffmpeg_temp_targets[0].parent.exists())
            self.assertEqual(list(temp_dir.iterdir()), [])
            self.assertEqual(result.file_path.parent, out_dir.resolve())
            self.assertEqual(len(list(out_dir.iterdir())), 1)

    def test_video_download_fallback_format_without_ffmpeg(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            calls: list[dict[str, Any]] = []
            downloader = MediaDownloader(
                ydl_factory=lambda opts: FakeYoutubeDL(opts, record_calls=calls),
                ffmpeg_checker=lambda: None,
            )

            result = downloader.download(
                DownloadRequest(
                    url="https://example.com/watch?v=123",
                    mode=DownloadMode.VIDEO,
                    output_dir=Path(tmp),
                )
            )
            self.assertTrue(result.file_path.exists())
            dl_opts = calls[-1]["options"]
            self.assertEqual(dl_opts["format"], "best[ext=mp4]/best")
            self.assertNotIn("merge_output_format", dl_opts)

    def test_valid_audio_download_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            calls: list[dict[str, Any]] = []
            downloader = MediaDownloader(
                ydl_factory=lambda opts: FakeYoutubeDL(
                    opts,
                    title="한국어 팟캐스트",
                    media_id="aud001",
                    ext="webm",
                    record_calls=calls,
                ),
                ffmpeg_checker=lambda: "C:/ffmpeg/bin/ffmpeg.exe",
            )

            result = downloader.download(
                DownloadRequest(
                    url="https://www.youtube.com/watch?v=aud001",
                    mode=DownloadMode.AUDIO,
                    output_dir=Path(tmp),
                    audio_format="mp3",
                )
            )

            self.assertEqual(result.mode, DownloadMode.AUDIO)
            self.assertTrue(result.file_path.exists())
            self.assertEqual(result.file_path.suffix, ".mp3")
            dl_opts = calls[-1]["options"]
            self.assertEqual(dl_opts["format"], "bestaudio/best")
            self.assertEqual(
                dl_opts["postprocessors"],
                [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}],
            )

    def test_audio_download_fails_cleanly_without_ffmpeg(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            events: list[DownloadProgress] = []
            downloader = MediaDownloader(
                ydl_factory=lambda opts: FakeYoutubeDL(opts),
                ffmpeg_checker=lambda: None,
            )

            with self.assertRaises(ExternalToolError):
                downloader.download(
                    DownloadRequest(
                        url="https://www.youtube.com/watch?v=aud001",
                        mode=DownloadMode.AUDIO,
                        output_dir=Path(tmp),
                    ),
                    progress_callback=events.append,
                )
            self.assertEqual(events[-1].status, DownloadStatus.ERROR)

    def test_existing_file_skip_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            existing = out_dir / "한국어 강의 샘플 [abc123XYZ].mp4"
            existing.write_bytes(b"existing-content")

            calls: list[dict[str, Any]] = []
            events: list[DownloadProgress] = []
            downloader = MediaDownloader(
                ydl_factory=lambda opts: FakeYoutubeDL(opts, record_calls=calls),
                ffmpeg_checker=lambda: "ffmpeg",
            )

            result = downloader.download(
                DownloadRequest(
                    url="https://www.youtube.com/watch?v=abc123XYZ",
                    mode=DownloadMode.VIDEO,
                    output_dir=out_dir,
                    existing_file_policy=ExistingFilePolicy.SKIP,
                ),
                progress_callback=events.append,
            )

            self.assertTrue(result.already_exists)
            self.assertEqual(result.file_path, existing.resolve())
            self.assertEqual(existing.read_bytes(), b"existing-content")
            # Only metadata probe (download=False) should have run
            self.assertEqual(len(calls), 1)
            self.assertFalse(calls[0]["download"])
            self.assertEqual(events[-1].status, DownloadStatus.SKIPPED)

    def test_existing_file_fail_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            existing = out_dir / "한국어 강의 샘플 [abc123XYZ].m4a"
            existing.write_bytes(b"existing-audio")

            downloader = MediaDownloader(
                ydl_factory=lambda opts: FakeYoutubeDL(opts, ext="webm"),
                ffmpeg_checker=lambda: "ffmpeg",
            )

            with self.assertRaises(FileAlreadyExistsError):
                downloader.download(
                    DownloadRequest(
                        url="https://www.youtube.com/watch?v=abc123XYZ",
                        mode=DownloadMode.AUDIO,
                        output_dir=out_dir,
                        existing_file_policy=ExistingFilePolicy.FAIL,
                        audio_format="m4a",
                    )
                )

    def test_existing_file_overwrite_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            existing = out_dir / "한국어 강의 샘플 [abc123XYZ].mp4"
            existing.write_bytes(b"old-bytes")

            calls: list[dict[str, Any]] = []
            downloader = MediaDownloader(
                ydl_factory=lambda opts: FakeYoutubeDL(
                    opts, payload=b"brand-new-bytes", record_calls=calls
                ),
                ffmpeg_checker=lambda: "ffmpeg",
            )

            result = downloader.download(
                DownloadRequest(
                    url="https://www.youtube.com/watch?v=abc123XYZ",
                    mode=DownloadMode.VIDEO,
                    output_dir=out_dir,
                    existing_file_policy=ExistingFilePolicy.OVERWRITE,
                )
            )

            self.assertFalse(result.already_exists)
            self.assertEqual(existing.read_bytes(), b"brand-new-bytes")
            self.assertEqual(len(calls), 1)
            self.assertTrue(calls[0]["download"])
            self.assertTrue(calls[0]["options"]["overwrites"])

    def test_network_connection_failures_map_to_network_error(self) -> None:
        network_exceptions = [
            DownloadError("ERROR: Unable to download webpage: <urlopen error [Errno 11001] getaddrinfo failed>"),
            DownloadError("ERROR: timed out while reading response"),
            URLError("Temporary failure in name resolution"),
            TimeoutError("Connection timed out"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            for exc in network_exceptions:
                with self.subTest(exc=type(exc).__name__):
                    downloader = MediaDownloader(
                        ydl_factory=lambda opts, e=exc: FakeYoutubeDL(
                            opts, error_on_extract=e
                        ),
                        ffmpeg_checker=lambda: "ffmpeg",
                    )
                    with self.assertRaises(NetworkDownloadError):
                        downloader.download(
                            DownloadRequest(
                                url="https://www.youtube.com/watch?v=netfail",
                                output_dir=Path(tmp),
                            )
                        )

    def test_download_interrupted_or_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # Case 1: KeyboardInterrupt mid-download
            downloader_kb = MediaDownloader(
                ydl_factory=lambda opts: FakeYoutubeDL(
                    opts, error_on_download=KeyboardInterrupt()
                ),
                ffmpeg_checker=lambda: "ffmpeg",
            )
            with self.assertRaises(DownloadInterruptedError):
                downloader_kb.download(
                    DownloadRequest(
                        url="https://www.youtube.com/watch?v=kb",
                        output_dir=Path(tmp),
                    )
                )

            # Case 2: Incomplete read / fragment interruption
            downloader_frag = MediaDownloader(
                ydl_factory=lambda opts: FakeYoutubeDL(
                    opts,
                    error_on_download=DownloadError(
                        "ERROR: Did not get any data blocks (interrupted)"
                    ),
                ),
                ffmpeg_checker=lambda: "ffmpeg",
            )
            with self.assertRaises(DownloadInterruptedError):
                downloader_frag.download(
                    DownloadRequest(
                        url="https://www.youtube.com/watch?v=frag",
                        output_dir=Path(tmp),
                    )
                )

            # Case 3: Post-processing / FFmpeg external tool failure
            downloader_pp = MediaDownloader(
                ydl_factory=lambda opts: FakeYoutubeDL(
                    opts,
                    error_on_download=PostProcessingError("ffmpeg exited with code 1"),
                ),
                ffmpeg_checker=lambda: "ffmpeg",
            )
            with self.assertRaises(ExternalToolError):
                downloader_pp.download(
                    DownloadRequest(
                        url="https://www.youtube.com/watch?v=pp",
                        output_dir=Path(tmp),
                    )
                )

            # Case 4: Generic unavailable video DownloadError
            downloader_unavail = MediaDownloader(
                ydl_factory=lambda opts: FakeYoutubeDL(
                    opts,
                    error_on_extract=DownloadError("ERROR: Video unavailable"),
                ),
                ffmpeg_checker=lambda: "ffmpeg",
            )
            with self.assertRaises(MediaDownloadError):
                downloader_unavail.download(
                    DownloadRequest(
                        url="https://www.youtube.com/watch?v=private",
                        output_dir=Path(tmp),
                    )
                )

            # Case 5: Extractor reports unsupported URL
            downloader_unsup = MediaDownloader(
                ydl_factory=lambda opts: FakeYoutubeDL(
                    opts,
                    error_on_extract=DownloadError("ERROR: Unsupported URL: https://example.com/nope"),
                ),
                ffmpeg_checker=lambda: "ffmpeg",
            )
            with self.assertRaises(InvalidURLError):
                downloader_unsup.download(
                    DownloadRequest(
                        url="https://example.com/nope",
                        output_dir=Path(tmp),
                    )
                )

            # Case 6: Download completes without producing file on disk
            downloader_missing_file = MediaDownloader(
                ydl_factory=lambda opts: FakeYoutubeDL(
                    opts, skip_file_creation=True
                ),
                ffmpeg_checker=lambda: "ffmpeg",
            )
            with self.assertRaises(MediaDownloadError):
                downloader_missing_file.download(
                    DownloadRequest(
                        url="https://www.youtube.com/watch?v=ghost",
                        output_dir=Path(tmp),
                    )
                )

    def test_invalid_audio_format_raises_media_download_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            downloader = MediaDownloader(
                ydl_factory=lambda opts: FakeYoutubeDL(opts),
                ffmpeg_checker=lambda: "ffmpeg",
            )
            with self.assertRaises(MediaDownloadError):
                downloader.download(
                    DownloadRequest(
                        url="https://www.youtube.com/watch?v=aud001",
                        mode=DownloadMode.AUDIO,
                        output_dir=Path(tmp),
                        audio_format="bad/format!",
                    )
                )

    def test_real_yt_dlp_accepts_generated_options(self) -> None:
        """Smoke test verifying yt_dlp.YoutubeDL accepts our option dict without error."""
        import yt_dlp

        with tempfile.TemporaryDirectory() as tmp:
            downloader = MediaDownloader()
            for mode in (DownloadMode.VIDEO, DownloadMode.AUDIO):
                req = DownloadRequest(
                    url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                    mode=mode,
                    output_dir=Path(tmp),
                )
                opts = downloader._build_ydl_options(
                    request=req,
                    output_dir=Path(tmp),
                    has_ffmpeg=True,
                    logger=downloader._build_ydl_options.__globals__["_SilentLogger"](),
                    progress_callback=None,
                    tracker={},
                )
                with yt_dlp.YoutubeDL(opts) as ydl:
                    prepared = ydl.prepare_filename(
                        {"title": "테스트 영상", "id": "vid01", "ext": "mp4"}
                    )
                    self.assertTrue(prepared.endswith("테스트 영상 [vid01].mp4"))


if __name__ == "__main__":
    unittest.main()
