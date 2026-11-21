"""Unit tests for centralized SubtitleUI path configuration and directory helpers."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from subtitleui.paths import (  # noqa: E402
    AUDIO_OUTPUT_DIR,
    CONFIG_DIR,
    DOWNLOAD_DIR,
    MODELS_DIR,
    NLLB_MODEL_DIR,
    OUTPUT_DIR,
    PROJECT_ROOT,
    TEMP_DIR,
    TRANSCRIPTION_DIR,
    TRANSLATION_DIR,
    default_audio_output_path,
    default_transcript_output_path,
    default_translation_output_path,
    ensure_dir,
    resolve_media_input_path,
    resolve_transcript_input_path,
    transcript_stem,
)


class TestPaths(unittest.TestCase):
    """Tests for centralized project directories and path resolution helpers."""

    def test_standard_directory_layout(self) -> None:
        self.assertEqual(PROJECT_ROOT, ROOT_DIR.resolve())
        self.assertEqual(DOWNLOAD_DIR, PROJECT_ROOT / "downloads")
        self.assertEqual(OUTPUT_DIR, PROJECT_ROOT / "outputs")
        self.assertEqual(AUDIO_OUTPUT_DIR, OUTPUT_DIR / "audio")
        self.assertEqual(TRANSCRIPTION_DIR, OUTPUT_DIR / "transcription")
        self.assertEqual(TEMP_DIR, PROJECT_ROOT / "temp")
        self.assertEqual(MODELS_DIR, PROJECT_ROOT / "models")

    def test_default_transcript_output_path_targets_outputs_transcription(self) -> None:
        sample_video = DOWNLOAD_DIR / "Wait, you guys knew each other？ [wTXo0uObVaE].mp4"
        out_path = default_transcript_output_path(sample_video)

        self.assertEqual(out_path.parent, TRANSCRIPTION_DIR)
        self.assertEqual(
            out_path.name,
            "Wait, you guys knew each other？ [wTXo0uObVaE].transcript.json",
        )
        self.assertNotEqual(out_path.parent, DOWNLOAD_DIR)

    def test_default_audio_output_path_targets_outputs_audio(self) -> None:
        sample_video = DOWNLOAD_DIR / "한국어 강의 [vid01].mp4"
        out_path = default_audio_output_path(sample_video)

        self.assertEqual(out_path.parent, AUDIO_OUTPUT_DIR)
        self.assertEqual(out_path.name, "한국어 강의 [vid01].wav")
        self.assertNotEqual(out_path.parent, DOWNLOAD_DIR)

    def test_resolve_media_input_path_falls_back_to_download_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fake_downloads = Path(tmp) / "downloads"
            fake_downloads.mkdir()
            video = fake_downloads / "sample clip [id123].mp4"
            video.write_bytes(b"video")

            resolved_by_name = resolve_media_input_path(
                "sample clip [id123].mp4",
                download_dir=fake_downloads,
            )
            self.assertEqual(resolved_by_name, video.resolve())

            resolved_by_full = resolve_media_input_path(
                video,
                download_dir=fake_downloads,
            )
            self.assertEqual(resolved_by_full, video.resolve())

    def test_ensure_dir_creates_missing_nested_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            nested = Path(tmp) / "outputs" / "transcription"
            self.assertFalse(nested.exists())
            created = ensure_dir(nested)
            self.assertTrue(created.exists())
            self.assertTrue(created.is_dir())

    def test_translation_path_helpers(self) -> None:
        self.assertEqual(CONFIG_DIR, PROJECT_ROOT / "config")
        self.assertEqual(TRANSLATION_DIR, OUTPUT_DIR / "translation")
        self.assertEqual(NLLB_MODEL_DIR, MODELS_DIR / "nllb")

        transcript_file = TRANSCRIPTION_DIR / "Wait, you guys knew each other？ [wTXo0uObVaE].transcript.json"
        self.assertEqual(
            transcript_stem(transcript_file),
            "Wait, you guys knew each other？ [wTXo0uObVaE]",
        )
        out_path = default_translation_output_path(transcript_file)
        self.assertEqual(out_path.parent, TRANSLATION_DIR)
        self.assertEqual(
            out_path.name,
            "Wait, you guys knew each other？ [wTXo0uObVaE].zh-TW.translation.json",
        )

        with tempfile.TemporaryDirectory() as tmp:
            fake_tr_dir = Path(tmp) / "outputs" / "transcription"
            fake_tr_dir.mkdir(parents=True)
            sample = fake_tr_dir / "sample.transcript.json"
            sample.write_text("{}", encoding="utf-8")
            resolved = resolve_transcript_input_path(
                "sample.transcript.json", transcription_dir=fake_tr_dir
            )
            self.assertEqual(resolved, sample.resolve())


if __name__ == "__main__":
    unittest.main()
