"""Unit tests for the SubtitleUI Phase 1 CLI interface."""

from __future__ import annotations

import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from yt_dlp.utils import DownloadError

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from subtitleui.asr import (  # noqa: E402
    CancellationToken,
    ModelConfig,
    TranscriptionEngine,
    load_transcript_json,
)
from subtitleui.cli import main  # noqa: E402
from subtitleui.downloader import MediaDownloader  # noqa: E402
from subtitleui.translator import (  # noqa: E402
    TranslationConfig,
    TranslationEngine,
    load_translation_json,
)
from tests.test_asr import FakeWhisperModel, _FakeRawSegment, _make_mock_extractor  # noqa: E402
from tests.test_downloader import FakeYoutubeDL  # noqa: E402
from tests.test_translator import _FakeBackend, _write_sample_asr_json  # noqa: E402


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


class TestTranscribeCLI(unittest.TestCase):
    """Tests for Phase 2 CLI local ASR transcription subcommand."""

    def test_cli_transcribe_full_audio_and_custom_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            media_file = Path(tmp) / "korean_clip.mp4"
            media_file.write_bytes(b"fake-media")
            out_json = Path(tmp) / "custom_range.json"
            stdout = io.StringIO()
            stderr = io.StringIO()

            def fake_engine_factory(cfg: ModelConfig) -> TranscriptionEngine:
                return TranscriptionEngine(
                    cfg,
                    audio_extractor=_make_mock_extractor(source_duration=1200.0),
                    model_factory=lambda name, **kw: FakeWhisperModel(
                        segments=[_FakeRawSegment(2.5, 6.8, " 오늘은 날씨가 정말 좋네요.")]
                    ),
                    cuda_checker=lambda: (True, None),
                )

            exit_code = main(
                [
                    "transcribe",
                    str(media_file),
                    "--model",
                    "medium",
                    "--start",
                    "00:12:30",
                    "--end",
                    "00:18:00",
                    "-o",
                    str(out_json),
                    "--model-dir",
                    str(Path(tmp) / "models"),
                ],
                engine_factory=fake_engine_factory,
                stdout=stdout,
                stderr=stderr,
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(stderr.getvalue(), "")
            output = stdout.getvalue()
            self.assertIn("辨識摘要", output)
            self.assertIn("00:12:30 ~ 00:18:00", output)
            self.assertIn("오늘은 날씨가 정말 좋네요.", output)

            loaded = load_transcript_json(out_json)
            self.assertEqual(loaded.metadata.range_start, 750.0)
            self.assertEqual(loaded.metadata.range_end, 1080.0)
            self.assertEqual(loaded.segments[0].start, 752.5)
            self.assertEqual(loaded.segments[0].end, 756.8)

    def test_cli_transcribe_interactive_custom_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            media_file = Path(tmp) / "sample.m4a"
            media_file.write_bytes(b"fake-audio")
            out_json = Path(tmp) / "interactive.json"
            stdout = io.StringIO()
            stderr = io.StringIO()
            responses = iter(
                [str(media_file), "small", "2", "00:01:00", "00:02:00", str(out_json)]
            )

            def fake_engine_factory(cfg: ModelConfig) -> TranscriptionEngine:
                return TranscriptionEngine(
                    cfg,
                    audio_extractor=_make_mock_extractor(source_duration=300.0),
                    model_factory=lambda name, **kw: FakeWhisperModel(
                        segments=[_FakeRawSegment(1.0, 4.0, " 안녕하세요.")]
                    ),
                    cuda_checker=lambda: (False, "cpu only"),
                )

            exit_code = main(
                ["transcribe", "--model-dir", str(Path(tmp) / "models")],
                engine_factory=fake_engine_factory,
                input_func=lambda: next(responses),
                stdout=stdout,
                stderr=stderr,
            )

            self.assertEqual(exit_code, 0)
            loaded = load_transcript_json(out_json)
            self.assertEqual(loaded.metadata.model, "small")
            self.assertEqual(loaded.segments[0].start, 61.0)
            self.assertEqual(loaded.segments[0].end, 64.0)

    def test_cli_transcribe_handles_invalid_range_and_cancellation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            media_file = Path(tmp) / "short.mp4"
            media_file.write_bytes(b"fake")

            def fake_engine_factory(cfg: ModelConfig) -> TranscriptionEngine:
                return TranscriptionEngine(
                    cfg,
                    audio_extractor=_make_mock_extractor(source_duration=60.0),
                    model_factory=lambda name, **kw: FakeWhisperModel(),
                    cuda_checker=lambda: (False, "cpu"),
                )

            # Invalid time range
            stderr_err = io.StringIO()
            code_err = main(
                [
                    "transcribe",
                    str(media_file),
                    "--start",
                    "00:00:30",
                    "--end",
                    "00:05:00",
                ],
                engine_factory=fake_engine_factory,
                stdout=io.StringIO(),
                stderr=stderr_err,
            )
            self.assertEqual(code_err, 1)
            self.assertIn("辨識失敗", stderr_err.getvalue())

            # Cancelled transcription
            token = CancellationToken()
            token.cancel()
            stderr_cancel = io.StringIO()
            code_cancel = main(
                ["transcribe", str(media_file)],
                engine_factory=fake_engine_factory,
                cancel_token=token,
                stdout=io.StringIO(),
                stderr=stderr_cancel,
            )
            self.assertEqual(code_cancel, 130)
            self.assertIn("辨識已取消", stderr_cancel.getvalue())

    def test_cli_default_download_and_transcription_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fake_downloads = Path(tmp) / "downloads"
            fake_transcriptions = Path(tmp) / "outputs" / "transcription"
            fake_temp = Path(tmp) / "temp"

            # Directories do not exist yet; both phases must auto-create them.
            self.assertFalse(fake_downloads.exists())
            self.assertFalse(fake_transcriptions.exists())

            with (
                mock.patch("subtitleui.cli.DOWNLOAD_DIR", fake_downloads),
                mock.patch("subtitleui.paths.DOWNLOAD_DIR", fake_downloads),
                mock.patch("subtitleui.paths.TRANSCRIPTION_DIR", fake_transcriptions),
            ):
                # Phase 1: default download goes into downloads/
                downloader = MediaDownloader(
                    ydl_factory=lambda opts: FakeYoutubeDL(
                        opts,
                        title="한국어 영상",
                        media_id="vid999",
                    ),
                    ffmpeg_checker=lambda: "ffmpeg",
                    temp_dir=fake_temp,
                )
                dl_out = io.StringIO()
                dl_err = io.StringIO()
                dl_code = main(
                    ["https://www.youtube.com/watch?v=vid999"],
                    downloader=downloader,
                    stdout=dl_out,
                    stderr=dl_err,
                )
                self.assertEqual(dl_code, 0, dl_err.getvalue())
                expected_video = fake_downloads / "한국어 영상 [vid999].mp4"
                self.assertTrue(expected_video.exists())

                # Phase 2: reads video from downloads/ (by filename) and writes JSON
                # by default into outputs/transcription/, NOT downloads/
                def fake_engine_factory(cfg: ModelConfig) -> TranscriptionEngine:
                    extractor = _make_mock_extractor(source_duration=600.0)
                    extractor._download_dir = fake_downloads
                    extractor._temp_dir = fake_temp
                    return TranscriptionEngine(
                        cfg,
                        audio_extractor=extractor,
                        model_factory=lambda name, **kw: FakeWhisperModel(
                            segments=[_FakeRawSegment(1.5, 4.2, " 안녕하세요.")]
                        ),
                        cuda_checker=lambda: (False, "cpu"),
                    )

                asr_out = io.StringIO()
                asr_err = io.StringIO()
                asr_code = main(
                    [
                        "transcribe",
                        "한국어 영상 [vid999].mp4",
                        "--start",
                        "00:01:00",
                        "--end",
                        "00:02:00",
                        "--model-dir",
                        str(Path(tmp) / "models"),
                    ],
                    engine_factory=fake_engine_factory,
                    stdout=asr_out,
                    stderr=asr_err,
                )
                self.assertEqual(asr_code, 0, asr_err.getvalue())

                expected_json = fake_transcriptions / "한국어 영상 [vid999].transcript.json"
                self.assertTrue(expected_json.exists())
                # Ensure NO JSON or temp file was written into downloads/
                self.assertEqual(
                    [p.name for p in fake_downloads.iterdir()],
                    ["한국어 영상 [vid999].mp4"],
                )
                # Ensure original video timestamps are preserved
                loaded = load_transcript_json(expected_json)
                self.assertEqual(loaded.metadata.source, str(expected_video.resolve()))
                self.assertEqual(loaded.metadata.range_start, 60.0)
                self.assertEqual(loaded.metadata.range_end, 120.0)
                self.assertEqual(loaded.segments[0].start, 61.5)
                self.assertEqual(loaded.segments[0].end, 64.2)


class TestTranslateCLI(unittest.TestCase):
    """Tests for Phase 3 CLI translate subcommand."""

    def test_cli_translate_full_segments_and_time_range_with_merge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            asr_path = _write_sample_asr_json(Path(tmp) / "clip.transcript.json")
            cfg_path = Path(tmp) / "config.yaml"
            out_path = Path(tmp) / "clip.zh-TW.translation.json"

            def fake_tr_factory(cfg: TranslationConfig) -> TranslationEngine:
                return TranslationEngine(
                    cfg,
                    backend=_FakeBackend(name=cfg.provider, model="test-model"),
                )

            # 1) Translate specific segments 1,3
            stdout = io.StringIO()
            stderr = io.StringIO()
            code = main(
                [
                    "translate",
                    str(asr_path),
                    "--segments",
                    "1,3",
                    "--backend",
                    "local-llm",
                    "--config",
                    str(cfg_path),
                    "-o",
                    str(out_path),
                ],
                translation_engine_factory=fake_tr_factory,
                stdout=stdout,
                stderr=stderr,
            )
            self.assertEqual(code, 0, stderr.getvalue())
            self.assertTrue(cfg_path.exists())
            self.assertIn("翻譯摘要", stdout.getvalue())
            loaded = load_translation_json(out_path)
            self.assertEqual([s.index for s in loaded.segments], [0, 2])

            # 2) Translate time range 00:00:15 ~ 00:00:24 and verify merge preserves 0, 2
            stdout2 = io.StringIO()
            stderr2 = io.StringIO()
            code2 = main(
                [
                    "translate",
                    str(asr_path),
                    "--start",
                    "00:00:15",
                    "--end",
                    "00:00:24",
                    "--config",
                    str(cfg_path),
                    "-o",
                    str(out_path),
                ],
                translation_engine_factory=fake_tr_factory,
                stdout=stdout2,
                stderr=stderr2,
            )
            self.assertEqual(code2, 0, stderr2.getvalue())
            loaded2 = load_translation_json(out_path)
            self.assertEqual([s.index for s in loaded2.segments], [0, 2, 4, 5])

    def test_cli_translate_handles_errors_and_cancellation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            asr_path = _write_sample_asr_json(Path(tmp) / "clip.transcript.json")
            cfg_path = Path(tmp) / "config.yaml"

            def fake_tr_factory(cfg: TranslationConfig) -> TranslationEngine:
                return TranslationEngine(cfg, backend=_FakeBackend())

            # Cancelled via token
            token = CancellationToken()
            token.cancel()
            stderr_cancel = io.StringIO()
            code_cancel = main(
                ["translate", str(asr_path), "--config", str(cfg_path)],
                translation_engine_factory=fake_tr_factory,
                cancel_token=token,
                stdout=io.StringIO(),
                stderr=stderr_cancel,
            )
            self.assertEqual(code_cancel, 130)
            self.assertIn("翻譯已取消", stderr_cancel.getvalue())

            # Conflicting selectors -> exit 1
            stderr_err = io.StringIO()
            code_err = main(
                [
                    "translate",
                    str(asr_path),
                    "--segments",
                    "1",
                    "--start",
                    "00:00:10",
                    "--config",
                    str(cfg_path),
                ],
                translation_engine_factory=fake_tr_factory,
                stdout=io.StringIO(),
                stderr=stderr_err,
            )
            self.assertEqual(code_err, 1)
            self.assertIn("翻譯失敗", stderr_err.getvalue())


if __name__ == "__main__":
    unittest.main()
