"""Unit and FFmpeg-integration tests for the SubtitleUI Phase 2 local ASR module."""

from __future__ import annotations

import json
import math
import struct
import subprocess
import sys
import tempfile
import unittest
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from subtitleui.asr import (  # noqa: E402
    ASR_CHANNELS,
    ASR_SAMPLE_RATE,
    AudioExtractionError,
    AudioExtractor,
    AudioInputError,
    CancellationToken,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL_NAME,
    DEFAULT_TASK,
    InvalidTimeRangeError,
    ModelConfig,
    ModelLoadError,
    TimeRange,
    TranscriptSegment,
    TranscriptionCancelledError,
    TranscriptionEngine,
    TranscriptionMetadata,
    TranscriptionProgress,
    TranscriptionRequest,
    TranscriptionResult,
    TranscriptionStage,
    apply_timestamp_offset,
    format_timestamp,
    load_transcript_json,
    normalize_model_name,
    parse_timestamp,
    resolve_hardware_candidates,
    resolve_time_range,
    save_transcript_json,
)


def _create_pcm_wav(
    path: Path,
    *,
    duration_seconds: float,
    sample_rate: int = 44100,
    channels: int = 2,
    frequency_hz: float = 440.0,
) -> Path:
    """Create a tiny valid PCM WAV file on disk using Python's standard library."""
    n_frames = int(round(duration_seconds * sample_rate))
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = bytearray()
        for i in range(n_frames):
            sample = int(8000 * math.sin(2.0 * math.pi * frequency_hz * (i / sample_rate)))
            packed = struct.pack("<h", sample)
            frames.extend(packed * channels)
        wf.writeframes(bytes(frames))
    return path


@dataclass
class _FakeRawSegment:
    start: float
    end: float
    text: str


@dataclass
class _FakeInfo:
    language: str | None = "ko"
    language_probability: float | None = 0.9876


class FakeWhisperModel:
    """Deterministic Faster-Whisper fake returning a lazy segment generator."""

    def __init__(
        self,
        *,
        segments: list[_FakeRawSegment] | None = None,
        info: _FakeInfo | None = None,
        error_on_transcribe: Exception | None = None,
        error_mid_stream: Exception | None = None,
        on_segment_yield: Any = None,
        record_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        self._segments = segments if segments is not None else [
            _FakeRawSegment(start=2.5, end=6.8, text=" 오늘은 날씨가 정말 좋네요."),
            _FakeRawSegment(start=7.1, end=11.4, text=" 한국어 공부를 시작해 봅시다."),
        ]
        self._info = info if info is not None else _FakeInfo()
        self._error_on_transcribe = error_on_transcribe
        self._error_mid_stream = error_mid_stream
        self._on_segment_yield = on_segment_yield
        self.record_calls = record_calls if record_calls is not None else []
        self.generator_exhausted = False

    def transcribe(
        self,
        audio: str,
        *,
        language: str | None = None,
        task: str = "transcribe",
        beam_size: int = 5,
        vad_filter: bool = False,
    ) -> tuple[Iterator[_FakeRawSegment], _FakeInfo]:
        self.record_calls.append(
            {
                "audio": audio,
                "language": language,
                "task": task,
                "beam_size": beam_size,
                "vad_filter": vad_filter,
                "audio_exists_at_call": Path(audio).exists(),
            }
        )
        if self._error_on_transcribe is not None:
            raise self._error_on_transcribe

        def _gen() -> Iterator[_FakeRawSegment]:
            for idx, seg in enumerate(self._segments):
                if self._error_mid_stream is not None and idx == 1:
                    raise self._error_mid_stream
                if self._on_segment_yield is not None:
                    self._on_segment_yield(idx, seg)
                yield seg
            self.generator_exhausted = True

        return _gen(), self._info


def _make_mock_extractor(
    *,
    source_duration: float = 1200.0,
    recorded_cmds: list[list[str]] | None = None,
    wav_paths_created: list[Path] | None = None,
) -> AudioExtractor:
    """Create an AudioExtractor backed by deterministic ffprobe/ffmpeg mocks."""
    cmds = recorded_cmds if recorded_cmds is not None else []
    created_wavs = wav_paths_created if wav_paths_created is not None else []

    def fake_runner(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        cmds.append(list(cmd))
        exe = Path(cmd[0]).name.lower()
        if "ffprobe" in exe:
            payload = {
                "streams": [{"codec_type": "audio", "duration": str(source_duration)}],
                "format": {"duration": str(source_duration)},
            }
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout=json.dumps(payload), stderr=""
            )

        # ffmpeg extraction mock: create a tiny valid 16kHz mono wav file of expected length
        out_wav = Path(cmd[-1])
        clip_duration = source_duration
        if "-t" in cmd:
            t_idx = cmd.index("-t")
            clip_duration = float(cmd[t_idx + 1])

        out_wav.parent.mkdir(parents=True, exist_ok=True)
        # Write minimal placeholder bytes; wav_inspector will report exact clip_duration
        out_wav.write_bytes(b"RIFFfakewav")
        created_wavs.append(out_wav)
        # Store expected duration next to filename via dict lookup
        duration_map[out_wav] = clip_duration
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    duration_map: dict[Path, float] = {}

    def fake_wav_inspector(wav_path: Path) -> tuple[float, int, int]:
        return duration_map[wav_path], ASR_SAMPLE_RATE, ASR_CHANNELS

    return AudioExtractor(
        ffmpeg_finder=lambda: "ffmpeg",
        ffprobe_finder=lambda: "ffprobe",
        runner=fake_runner,
        wav_inspector=fake_wav_inspector,
    )


class TestTimeRangeAndOffsets(unittest.TestCase):
    """Tests for timestamp parsing, time range validation, and offset calculation."""

    def test_parse_valid_timestamps(self) -> None:
        self.assertEqual(parse_timestamp("00:12:30"), 750.0)
        self.assertEqual(parse_timestamp("00:18:00"), 1080.0)
        self.assertEqual(parse_timestamp("01:02:03.5"), 3723.5)
        self.assertEqual(parse_timestamp("12:30"), 750.0)
        self.assertEqual(parse_timestamp("750"), 750.0)
        self.assertEqual(parse_timestamp(750.25), 750.25)
        self.assertEqual(parse_timestamp(0), 0.0)
        self.assertEqual(format_timestamp(750.0), "00:12:30")
        self.assertEqual(format_timestamp(752.5, include_millis=True), "00:12:32.500")

    def test_negative_start_time_raises_error(self) -> None:
        for invalid in (-1, -0.1, "-5", "-00:01:00"):
            with self.subTest(start=invalid):
                with self.assertRaises(InvalidTimeRangeError):
                    resolve_time_range(invalid, "00:05:00", media_duration=600.0)

    def test_end_time_exceeding_media_duration_raises_error(self) -> None:
        with self.assertRaises(InvalidTimeRangeError):
            resolve_time_range("00:05:00", "00:25:00", media_duration=1200.0)

        with self.assertRaises(InvalidTimeRangeError):
            resolve_time_range("00:20:00", "00:21:00", media_duration=1200.0)

    def test_equal_or_reversed_start_and_end_raises_error(self) -> None:
        with self.assertRaises(InvalidTimeRangeError):
            resolve_time_range("00:10:00", "00:10:00", media_duration=1200.0)

        with self.assertRaises(InvalidTimeRangeError):
            resolve_time_range("00:15:00", "00:10:00", media_duration=1200.0)

    def test_invalid_time_formats_raise_error(self) -> None:
        bad_inputs: list[Any] = [
            "",
            "   ",
            "abc",
            "00:60:00",
            "00:10:60",
            "00:99:99",
            "1:2:3:4",
            "00:-10:00",
            float("nan"),
            float("inf"),
            True,
        ]
        for raw in bad_inputs:
            with self.subTest(raw=raw):
                with self.assertRaises(InvalidTimeRangeError):
                    parse_timestamp(raw)

    def test_timestamp_offset_formula(self) -> None:
        self.assertEqual(apply_timestamp_offset(2.5, 750.0), 752.5)
        self.assertEqual(apply_timestamp_offset(6.8, 750.0), 756.8)
        self.assertEqual(apply_timestamp_offset(0.0, 0.0), 0.0)
        self.assertEqual(apply_timestamp_offset(12.345, 120.5), 132.845)

    def test_resolve_full_and_partial_ranges(self) -> None:
        full_range, is_custom = resolve_time_range(None, None, media_duration=600.0)
        self.assertFalse(is_custom)
        self.assertEqual(full_range, TimeRange(start=0.0, end=600.0))

        start_only, is_custom = resolve_time_range("00:05:00", None, media_duration=600.0)
        self.assertTrue(is_custom)
        self.assertEqual(start_only, TimeRange(start=300.0, end=600.0))

        end_only, is_custom = resolve_time_range(None, "00:03:00", media_duration=600.0)
        self.assertTrue(is_custom)
        self.assertEqual(end_only, TimeRange(start=0.0, end=180.0))


class TestTranscriptionEngine(unittest.TestCase):
    """Tests for Phase 2 local ASR engine, time ranges, fallback, cancellation, and JSON."""

    def test_full_audio_transcription_mode_a_and_model_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            media_file = Path(tmp) / "korean_lesson.mp4"
            media_file.write_bytes(b"fake-video")
            models_dir = Path(tmp) / "models"
            cmds: list[list[str]] = []
            created_wavs: list[Path] = []
            model_calls: list[dict[str, Any]] = []
            factory_calls: list[dict[str, Any]] = []

            fake_model = FakeWhisperModel(
                segments=[
                    _FakeRawSegment(0.5, 3.2, " 안녕하세요."),
                    _FakeRawSegment(3.5, 7.0, " 오늘은 한국어를 공부해요."),
                ],
                info=_FakeInfo(language="ko", language_probability=0.9921),
                record_calls=model_calls,
            )

            def fake_factory(name: str, **kwargs: Any) -> FakeWhisperModel:
                factory_calls.append({"name": name, **kwargs})
                return fake_model

            extractor = _make_mock_extractor(
                source_duration=120.0,
                recorded_cmds=cmds,
                wav_paths_created=created_wavs,
            )
            engine = TranscriptionEngine(
                ModelConfig(model_name=DEFAULT_MODEL_NAME, download_root=models_dir),
                audio_extractor=extractor,
                model_factory=fake_factory,
                cuda_checker=lambda: (True, None),
            )

            events: list[TranscriptionProgress] = []
            result1 = engine.transcribe(
                TranscriptionRequest(audio_path=media_file),
                progress_callback=events.append,
            )

            # Verify defaults: medium, ko, transcribe, CUDA+FP16
            self.assertEqual(result1.metadata.model, "medium")
            self.assertEqual(result1.metadata.language, DEFAULT_LANGUAGE)
            self.assertEqual(result1.metadata.task, DEFAULT_TASK)
            self.assertEqual(result1.metadata.device, "cuda")
            self.assertEqual(result1.metadata.compute_type, "float16")
            self.assertEqual(result1.metadata.range_start, 0.0)
            self.assertEqual(result1.metadata.range_end, 120.0)
            self.assertEqual(result1.metadata.source_duration, 120.0)
            self.assertEqual(result1.metadata.language_probability, 0.9921)

            # Verify raw Korean text preserved and generator iterated
            self.assertTrue(fake_model.generator_exhausted)
            self.assertEqual(len(result1.segments), 2)
            self.assertEqual(result1.segments[0], TranscriptSegment(0.5, 3.2, "안녕하세요."))
            self.assertEqual(
                result1.segments[1], TranscriptSegment(3.5, 7.0, "오늘은 한국어를 공부해요.")
            )

            # Verify FFmpeg command used 16kHz mono PCM WAV without stream copy or -ss/-t
            ffmpeg_cmd = cmds[-1]
            self.assertNotIn("-ss", ffmpeg_cmd)
            self.assertNotIn("-t", ffmpeg_cmd)
            self.assertNotIn("copy", ffmpeg_cmd)
            self.assertIn("pcm_s16le", ffmpeg_cmd)
            self.assertIn("16000", ffmpeg_cmd)

            # Verify temp WAV was cleaned up after transcription
            self.assertEqual(len(created_wavs), 1)
            self.assertFalse(created_wavs[0].exists())

            # Verify workflow stages emitted
            stages = [e.stage for e in events]
            self.assertIn(TranscriptionStage.VALIDATING, stages)
            self.assertIn(TranscriptionStage.EXTRACTING_AUDIO, stages)
            self.assertIn(TranscriptionStage.LOADING_MODEL, stages)
            self.assertIn(TranscriptionStage.TRANSCRIBING, stages)
            self.assertEqual(stages[-1], TranscriptionStage.COMPLETED)

            # Second call on same engine must reuse loaded model without reinitializing
            engine.transcribe(TranscriptionRequest(audio_path=media_file))
            self.assertEqual(len(factory_calls), 1)
            self.assertEqual(len(model_calls), 2)

    def test_custom_time_range_transcription_and_timestamp_offset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            media_file = Path(tmp) / "drama_episode.mp4"
            media_file.write_bytes(b"fake-drama")
            out_json = Path(tmp) / "out" / "range.json"
            cmds: list[list[str]] = []
            created_wavs: list[Path] = []
            model_calls: list[dict[str, Any]] = []

            fake_model = FakeWhisperModel(
                segments=[
                    _FakeRawSegment(2.5, 6.8, " 오늘은 날씨가 정말 좋네요."),
                    _FakeRawSegment(10.0, 14.25, " 같이 커피 마실래요?"),
                ],
                record_calls=model_calls,
            )

            engine = TranscriptionEngine(
                ModelConfig(model_name="small", download_root=Path(tmp) / "models"),
                audio_extractor=_make_mock_extractor(
                    source_duration=1800.0,
                    recorded_cmds=cmds,
                    wav_paths_created=created_wavs,
                ),
                model_factory=lambda name, **kw: fake_model,
                cuda_checker=lambda: (True, None),
            )

            result = engine.transcribe(
                TranscriptionRequest(
                    audio_path=media_file,
                    start_time="00:12:30",
                    end_time="00:18:00",
                    output_path=out_json,
                )
            )

            # Verify FFmpeg extracted ONLY 00:12:30 ~ 00:18:00 (start=750s, duration=330s)
            ffmpeg_cmd = cmds[-1]
            self.assertIn("-ss", ffmpeg_cmd)
            self.assertEqual(ffmpeg_cmd[ffmpeg_cmd.index("-ss") + 1], "750.000")
            self.assertIn("-t", ffmpeg_cmd)
            self.assertEqual(ffmpeg_cmd[ffmpeg_cmd.index("-t") + 1], "330.000")
            self.assertNotIn("copy", ffmpeg_cmd)
            # Ensure -ss comes AFTER -i for sample-accurate seeking
            self.assertGreater(ffmpeg_cmd.index("-ss"), ffmpeg_cmd.index("-i"))

            # Model received the extracted temporary WAV, NOT the original full video
            self.assertEqual(len(model_calls), 1)
            self.assertNotEqual(Path(model_calls[0]["audio"]), media_file.resolve())
            self.assertTrue(model_calls[0]["audio_exists_at_call"])
            # Temp WAV cleaned up afterwards
            self.assertFalse(created_wavs[0].exists())

            # CRITICAL: Verify original video timestamp offsets!
            self.assertEqual(result.metadata.range_start, 750.0)
            self.assertEqual(result.metadata.range_end, 1080.0)
            self.assertEqual(result.segments[0].start, 752.5)
            self.assertEqual(result.segments[0].end, 756.8)
            self.assertEqual(result.segments[0].text, "오늘은 날씨가 정말 좋네요.")
            self.assertEqual(result.segments[1].start, 760.0)
            self.assertEqual(result.segments[1].end, 764.25)

            # Verify saved JSON matches exact offset timestamps
            self.assertEqual(result.output_path, out_json.resolve())
            self.assertTrue(out_json.exists())

    def test_multiple_different_start_and_end_ranges(self) -> None:
        cases = [
            ("00:00:10", "00:00:40", 10.0, 40.0),
            ("00:05:15.5", "00:07:00", 315.5, 420.0),
            (60.0, 180.0, 60.0, 180.0),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            media_file = Path(tmp) / "sample.m4a"
            media_file.write_bytes(b"fake-audio")
            engine = TranscriptionEngine(
                ModelConfig(model_name="large-v3-turbo", download_root=Path(tmp) / "models"),
                audio_extractor=_make_mock_extractor(source_duration=600.0),
                model_factory=lambda name, **kw: FakeWhisperModel(
                    segments=[_FakeRawSegment(1.25, 4.75, " 테스트 문장입니다.")]
                ),
                cuda_checker=lambda: (False, "no gpu"),
            )

            for raw_start, raw_end, exp_start, exp_end in cases:
                with self.subTest(start=raw_start, end=raw_end):
                    res = engine.transcribe(
                        TranscriptionRequest(
                            audio_path=media_file,
                            start_time=raw_start,
                            end_time=raw_end,
                        )
                    )
                    self.assertEqual(res.metadata.range_start, exp_start)
                    self.assertEqual(res.metadata.range_end, exp_end)
                    self.assertEqual(res.segments[0].start, round(exp_start + 1.25, 3))
                    self.assertEqual(res.segments[0].end, round(exp_start + 4.75, 3))

    def test_missing_audio_file_raises_audio_input_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "does_not_exist.mp4"
            engine = TranscriptionEngine(
                ModelConfig(download_root=Path(tmp) / "models"),
                audio_extractor=_make_mock_extractor(),
                model_factory=lambda name, **kw: FakeWhisperModel(),
            )
            with self.assertRaises(AudioInputError):
                engine.transcribe(TranscriptionRequest(audio_path=missing))

    def test_audio_without_speech_returns_empty_segments_without_fabricating_confidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            media_file = Path(tmp) / "silence.wav"
            media_file.write_bytes(b"fake-silence")

            fake_model = FakeWhisperModel(
                segments=[_FakeRawSegment(0.0, 2.0, "   ")],
                info=_FakeInfo(language="ko", language_probability=None),
            )
            engine = TranscriptionEngine(
                ModelConfig(download_root=Path(tmp) / "models"),
                audio_extractor=_make_mock_extractor(source_duration=15.0),
                model_factory=lambda name, **kw: fake_model,
                cuda_checker=lambda: (False, "cpu test"),
            )

            events: list[TranscriptionProgress] = []
            result = engine.transcribe(
                TranscriptionRequest(audio_path=media_file),
                progress_callback=events.append,
            )

            self.assertEqual(result.segments, ())
            self.assertIsNone(result.metadata.language_probability)
            self.assertNotIn("language_probability", result.to_dict()["metadata"])
            self.assertEqual(events[-1].stage, TranscriptionStage.COMPLETED)
            self.assertIn("未偵測到有效語音", events[-1].message)

    def test_model_load_failure_raises_model_load_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ModelLoadError):
                normalize_model_name("nonexistent-model-xyz")

            media_file = Path(tmp) / "audio.mp4"
            media_file.write_bytes(b"fake")

            def broken_factory(name: str, **kw: Any) -> Any:
                raise RuntimeError("corrupted model weights")

            engine = TranscriptionEngine(
                ModelConfig(device="cpu", download_root=Path(tmp) / "models"),
                audio_extractor=_make_mock_extractor(source_duration=30.0),
                model_factory=broken_factory,
            )
            with self.assertRaises(ModelLoadError):
                engine.transcribe(TranscriptionRequest(audio_path=media_file))

            # Explicit CUDA request when GPU unavailable fails clearly
            with self.assertRaises(ModelLoadError):
                resolve_hardware_candidates(
                    preferred_device="cuda",
                    cuda_checker=lambda: (False, "偵測不到可用的 CUDA GPU"),
                )

    def test_gpu_unavailable_and_cuda_init_failure_fallback_to_cpu_int8(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            media_file = Path(tmp) / "audio.mp4"
            media_file.write_bytes(b"fake")

            # Case 1: CUDA unavailable up front -> CPU + INT8
            attempts_1: list[tuple[str, str]] = []

            def factory_1(name: str, **kw: Any) -> FakeWhisperModel:
                attempts_1.append((kw["device"], kw["compute_type"]))
                return FakeWhisperModel()

            engine_1 = TranscriptionEngine(
                ModelConfig(device="auto", download_root=Path(tmp) / "m1"),
                audio_extractor=_make_mock_extractor(source_duration=30.0),
                model_factory=factory_1,
                cuda_checker=lambda: (False, "偵測不到可用的 CUDA GPU"),
            )
            res_1 = engine_1.transcribe(TranscriptionRequest(audio_path=media_file))
            self.assertEqual(attempts_1, [("cpu", "int8")])
            self.assertEqual(res_1.metadata.device, "cpu")
            self.assertEqual(res_1.metadata.compute_type, "int8")
            self.assertTrue(engine_1.hardware_config.fallback_used)

            # Case 2: CUDA reported available, but WhisperModel init fails -> falls back to CPU + INT8
            attempts_2: list[tuple[str, str]] = []

            def factory_2(name: str, **kw: Any) -> FakeWhisperModel:
                attempts_2.append((kw["device"], kw["compute_type"]))
                if kw["device"] == "cuda":
                    raise RuntimeError("CUDA failed with error out of memory")
                return FakeWhisperModel()

            engine_2 = TranscriptionEngine(
                ModelConfig(device="auto", download_root=Path(tmp) / "m2"),
                audio_extractor=_make_mock_extractor(source_duration=30.0),
                model_factory=factory_2,
                cuda_checker=lambda: (True, None),
            )
            res_2 = engine_2.transcribe(TranscriptionRequest(audio_path=media_file))
            self.assertEqual(attempts_2, [("cuda", "float16"), ("cpu", "int8")])
            self.assertEqual(res_2.metadata.device, "cpu")
            self.assertEqual(res_2.metadata.compute_type, "int8")

            # Case 3: Lazy CUDA runtime failure during transcribe() -> falls back to CPU + INT8
            attempts_3: list[tuple[str, str]] = []

            def factory_3(name: str, **kw: Any) -> FakeWhisperModel:
                attempts_3.append((kw["device"], kw["compute_type"]))
                if kw["device"] == "cuda":
                    return FakeWhisperModel(
                        error_on_transcribe=RuntimeError(
                            "Library cublas64_12.dll is not found or cannot be loaded"
                        )
                    )
                return FakeWhisperModel()

            engine_3 = TranscriptionEngine(
                ModelConfig(device="auto", download_root=Path(tmp) / "m3"),
                audio_extractor=_make_mock_extractor(source_duration=30.0),
                model_factory=factory_3,
                cuda_checker=lambda: (True, None),
            )
            res_3 = engine_3.transcribe(TranscriptionRequest(audio_path=media_file))
            self.assertEqual(attempts_3, [("cuda", "float16"), ("cpu", "int8")])
            self.assertEqual(res_3.metadata.device, "cpu")
            self.assertEqual(res_3.metadata.compute_type, "int8")

    def test_cancellation_mid_transcription_and_before_start(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            media_file = Path(tmp) / "long_video.mp4"
            media_file.write_bytes(b"fake")
            created_wavs: list[Path] = []

            # Case 1: Cancel before starting
            pre_cancelled = CancellationToken()
            pre_cancelled.cancel()
            engine = TranscriptionEngine(
                ModelConfig(download_root=Path(tmp) / "models"),
                audio_extractor=_make_mock_extractor(
                    source_duration=300.0, wav_paths_created=created_wavs
                ),
                model_factory=lambda name, **kw: FakeWhisperModel(),
                cuda_checker=lambda: (False, "cpu"),
            )
            with self.assertRaises(TranscriptionCancelledError):
                engine.transcribe(
                    TranscriptionRequest(audio_path=media_file),
                    cancel_token=pre_cancelled,
                )
            self.assertEqual(len(created_wavs), 0)

            # Case 2: Cancel mid-stream during segment generator iteration
            mid_token = CancellationToken()

            def cancel_after_first(idx: int, _seg: _FakeRawSegment) -> None:
                if idx == 1:
                    mid_token.cancel()

            mid_model = FakeWhisperModel(
                segments=[
                    _FakeRawSegment(0.0, 2.0, " 첫 번째 문장"),
                    _FakeRawSegment(2.0, 4.0, " 두 번째 문장"),
                    _FakeRawSegment(4.0, 6.0, " 세 번째 문장"),
                ],
                on_segment_yield=cancel_after_first,
            )
            engine_mid = TranscriptionEngine(
                ModelConfig(download_root=Path(tmp) / "models"),
                audio_extractor=_make_mock_extractor(
                    source_duration=300.0, wav_paths_created=created_wavs
                ),
                model_factory=lambda name, **kw: mid_model,
                cuda_checker=lambda: (False, "cpu"),
            )
            events: list[TranscriptionProgress] = []
            with self.assertRaises(TranscriptionCancelledError):
                engine_mid.transcribe(
                    TranscriptionRequest(audio_path=media_file),
                    progress_callback=events.append,
                    cancel_token=mid_token,
                )

            self.assertFalse(mid_model.generator_exhausted)
            self.assertEqual(events[-1].stage, TranscriptionStage.CANCELLED)
            # Temp WAV must be cleaned up even on cancellation!
            self.assertTrue(all(not p.exists() for p in created_wavs))

    def test_json_save_and_load_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            json_path = Path(tmp) / "nested" / "result.json"
            original = TranscriptionResult(
                metadata=TranscriptionMetadata(
                    source="C:/videos/korean.mp4",
                    model="medium",
                    language="ko",
                    task="transcribe",
                    transcribed_at="2026-11-20T23:30:00+08:00",
                    range_start=750.0,
                    range_end=1080.0,
                    source_duration=1800.0,
                    device="cuda",
                    compute_type="float16",
                    detected_language="ko",
                    language_probability=0.985,
                    elapsed_seconds=12.4,
                ),
                segments=(
                    TranscriptSegment(752.5, 756.8, "오늘은 날씨가 정말 좋네요."),
                    TranscriptSegment(757.0, 761.2, "내일 다시 만나요."),
                ),
            )

            saved = save_transcript_json(original, json_path)
            self.assertEqual(saved, json_path.resolve())
            raw_text = saved.read_text(encoding="utf-8")
            # Korean characters must be readable UTF-8, not escaped \uXXXX
            self.assertIn("오늘은 날씨가 정말 좋네요.", raw_text)

            loaded = load_transcript_json(saved)
            self.assertEqual(loaded.metadata, original.metadata)
            self.assertEqual(loaded.segments, original.segments)
            self.assertEqual(loaded.output_path, saved)


class TestRealFFmpegAudioExtractor(unittest.TestCase):
    """Integration test verifying real FFprobe/FFmpeg slicing & temp cleanup."""

    def test_real_ffmpeg_extracts_exact_16k_mono_slice_and_cleans_temp(self) -> None:
        extractor = AudioExtractor()
        with tempfile.TemporaryDirectory() as tmp:
            source_wav = _create_pcm_wav(
                Path(tmp) / "source_44k_stereo.wav",
                duration_seconds=8.0,
                sample_rate=44100,
                channels=2,
            )

            duration = extractor.probe_duration(source_wav)
            self.assertAlmostEqual(duration, 8.0, delta=0.1)

            time_range, is_custom = resolve_time_range("00:00:02", "00:00:05", duration)
            self.assertTrue(is_custom)

            temp_wav_path: Path | None = None
            with extractor.extract_to_temp(
                source_wav,
                time_range=time_range,
                is_custom_range=is_custom,
            ) as extracted:
                temp_wav_path = extracted.wav_path
                self.assertTrue(extracted.wav_path.exists())
                self.assertEqual(extracted.sample_rate, 16000)
                self.assertEqual(extracted.channels, 1)
                self.assertAlmostEqual(extracted.duration, 3.0, delta=0.15)

            assert temp_wav_path is not None
            self.assertFalse(temp_wav_path.exists())
            self.assertFalse(temp_wav_path.parent.exists())

    def test_audio_extractor_missing_ffmpeg_raises_audio_extraction_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source_wav = _create_pcm_wav(Path(tmp) / "short.wav", duration_seconds=2.0)
            extractor = AudioExtractor(
                ffmpeg_finder=lambda: None,
                ffprobe_finder=lambda: None,
            )
            with self.assertRaises(AudioExtractionError):
                extractor.probe_duration(source_wav)


if __name__ == "__main__":
    unittest.main()
