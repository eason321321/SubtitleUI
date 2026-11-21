"""Comprehensive unit tests for SubtitleUI Phase 3 translation system."""

from __future__ import annotations

import json
import socket
import sys
import tempfile
import unittest
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest import mock

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from subtitleui.translator import (  # noqa: E402
    AlignmentError,
    BackendBatchItem,
    BackendBatchRequest,
    BackendTimeoutError,
    BackendUnavailableError,
    CT2NLLBBackend,
    CT2NLLBConfig,
    ConfigError,
    ConfigManager,
    ExistingTranslationPolicy,
    LocalLLMBackend,
    LocalLLMConfig,
    SelectionError,
    SelectionMode,
    SourceSegment,
    TranscriptReadError,
    TranslatedSegment,
    TranslationBackend,
    TranslationCancellationToken,
    TranslationCancelledError,
    TranslationConfig,
    TranslationEngine,
    TranslationProgress,
    TranslationRequest,
    TranslationStage,
    TranslationStorageError,
    build_translation_messages,
    load_asr_transcript,
    load_translation_json,
    parse_and_validate_llm_response,
    parse_segment_spec,
    resolve_api_key,
    save_translation_json,
    select_segments,
)


def _write_sample_asr_json(
    path: Path,
    segments: list[dict[str, Any]] | None = None,
) -> Path:
    default_segments = [
        {"start": 0.0, "end": 2.0, "text": "안녕하세요"},
        {"start": 2.5, "end": 5.0, "text": "오늘 날씨가 정말 좋네요"},
        {"start": 10.0, "end": 12.0, "text": "아니야 방금 굉장히 인위적이었어"},
        {"start": 12.0, "end": 13.5, "text": "카메라 없는거 아니야?"},
        {"start": 15.0, "end": 18.0, "text": "대박이다"},
        {"start": 20.0, "end": 24.0, "text": "내일 같이 운동하자"},
    ]
    payload = {
        "metadata": {
            "source": "C:\\videos\\sample.mp4",
            "model": "medium",
            "language": "ko",
            "task": "transcribe",
            "transcribed_at": "2026-11-21T13:00:00+08:00",
            "range_start": 0.0,
            "range_end": 24.0,
            "source_duration": 24.0,
            "device": "cuda",
            "compute_type": "float16",
        },
        "segments": default_segments if segments is None else segments,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


class _FakeNLLBTokenizer:
    def encode(self, text: str, src_lang: str) -> list[str]:
        return [src_lang, *text.split(), "</s>"]

    def decode(self, tokens: list[str], tgt_lang: str) -> str:
        filtered = [t for t in tokens if t not in (tgt_lang, "<s>", "</s>")]
        return " ".join(filtered)


@dataclass
class _FakeCT2Result:
    hypotheses: list[list[str]]


class _FakeCT2Translator:
    def __init__(self, *, fail_once_with: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._fail_once = fail_once_with

    def translate_batch(
        self,
        source: list[list[str]],
        *,
        target_prefix: list[list[str]],
        beam_size: int = 4,
        max_decoding_length: int = 256,
    ) -> list[_FakeCT2Result]:
        self.calls.append(
            {
                "source": source,
                "target_prefix": target_prefix,
                "beam_size": beam_size,
                "max_decoding_length": max_decoding_length,
            }
        )
        if self._fail_once is not None:
            exc = self._fail_once
            self._fail_once = None
            raise exc
        results: list[_FakeCT2Result] = []
        for tokens, prefix in zip(source, target_prefix):
            words = [t for t in tokens if t not in ("kor_Hang", "</s>")]
            results.append(_FakeCT2Result(hypotheses=[[prefix[0], f"譯:{' '.join(words)}"]]))
        return results


class _FakeBackend(TranslationBackend):
    """Deterministic fake backend for engine/CLI/retry tests."""

    def __init__(
        self,
        *,
        name: str = "local-llm",
        model: str = "fake-qwen3:4b",
        supports_context: bool = True,
        handler=None,
    ) -> None:
        self._name = name
        self._model = model
        self._supports_context = supports_context
        self._handler = handler
        self._loaded = False
        self.load_count = 0
        self.unload_count = 0
        self.requests: list[BackendBatchRequest] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def supports_context(self) -> bool:
        return self._supports_context

    def load(self) -> None:
        self._loaded = True
        self.load_count += 1

    def unload(self) -> None:
        self._loaded = False
        self.unload_count += 1

    def translate_batch(
        self,
        request: BackendBatchRequest,
        *,
        cancel_token=None,
    ) -> dict[int, str]:
        self.requests.append(request)
        if self._handler is not None:
            return self._handler(request)
        return {
            item.index: (f"繁中({item.text})" if item.text.strip() else "")
            for item in request.items
        }


class TestConfigManager(unittest.TestCase):
    """Tests for ConfigManager auto-creation, loading, validation, and secret lookup."""

    def test_auto_creates_default_config_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = ConfigManager(project_root=root)
            self.assertFalse(manager.config_path.exists())

            cfg = manager.load(auto_create=True)
            self.assertTrue(manager.config_path.exists())
            self.assertEqual(cfg.provider, "ct2-nllb")
            self.assertEqual(cfg.target_language, "zh-TW")
            self.assertEqual(cfg.batch_size, 10)
            self.assertEqual(cfg.context_before, 2)
            self.assertEqual(cfg.context_after, 2)
            self.assertEqual(cfg.local_llm.backend, "ollama")
            self.assertEqual(cfg.local_llm.base_url, "http://localhost:11434/v1")
            self.assertEqual(cfg.local_llm.model, "qwen3:4b")
            self.assertEqual(cfg.ct2_nllb.model_path, (root / "models" / "nllb").resolve())

    def test_invalid_yaml_raises_config_error_without_overwriting_user_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "config.yaml"
            broken_content = "translation:\n  provider: [unclosed\n"
            cfg_path.write_text(broken_content, encoding="utf-8")

            manager = ConfigManager(cfg_path)
            with self.assertRaises(ConfigError):
                manager.load(auto_create=True)

            # User file must remain untouched!
            self.assertEqual(cfg_path.read_text(encoding="utf-8"), broken_content)

    def test_invalid_provider_or_url_raises_clear_config_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "config.yaml"
            cfg_path.write_text(
                "translation:\n  provider: unknown-backend\n", encoding="utf-8"
            )
            with self.assertRaises(ConfigError) as ctx:
                ConfigManager(cfg_path).load()
            self.assertIn("provider", str(ctx.exception))

            cfg_path.write_text(
                "translation:\n  provider: local-llm\n  local_llm:\n    base_url: ftp://invalid\n",
                encoding="utf-8",
            )
            with self.assertRaises(ConfigError) as ctx2:
                ConfigManager(cfg_path).load()
            self.assertIn("base_url", str(ctx2.exception))

    def test_resolve_api_key_prefers_env_then_credential_store(self) -> None:
        class _FakeCredStore:
            def get_secret(self, key: str) -> str | None:
                return "cred-secret" if key == "subtitleui/llm" else None

        cfg_env = LocalLLMConfig(
            api_key="plaintext",
            api_key_env="SUBTITLEUI_TEST_API_KEY",
            credential_key="subtitleui/llm",
        )
        with mock.patch.dict("os.environ", {"SUBTITLEUI_TEST_API_KEY": "env-secret"}):
            self.assertEqual(
                resolve_api_key(cfg_env, credential_store=_FakeCredStore()),
                "env-secret",
            )

        cfg_cred = LocalLLMConfig(
            api_key="plaintext",
            api_key_env="",
            credential_key="subtitleui/llm",
        )
        self.assertEqual(
            resolve_api_key(cfg_cred, credential_store=_FakeCredStore()),
            "cred-secret",
        )


class TestSelectorAndStorage(unittest.TestCase):
    """Tests for segment/time selection, ASR reading, merge storage, and ASR protection."""

    def test_segment_spec_1_based_to_0_based(self) -> None:
        self.assertEqual(parse_segment_spec("1,3,5-6", 6), (0, 2, 4, 5))
        self.assertEqual(parse_segment_spec(" 4 , 2-3 , 2 ", 6), (1, 2, 3))

        with self.assertRaises(SelectionError):
            parse_segment_spec("0", 6)
        with self.assertRaises(SelectionError):
            parse_segment_spec("7", 6)
        with self.assertRaises(SelectionError):
            parse_segment_spec("5-2", 6)
        with self.assertRaises(SelectionError):
            parse_segment_spec("abc", 6)

    def test_time_range_intersection_rule(self) -> None:
        segments = (
            SourceSegment(0, 0.0, 10.0, "a"),  # end == start (10.0) -> NOT intersecting
            SourceSegment(1, 9.5, 11.0, "b"),  # intersects [10, 15]
            SourceSegment(2, 11.0, 14.0, "c"),  # fully inside
            SourceSegment(3, 14.5, 16.0, "d"),  # intersects [10, 15]
            SourceSegment(4, 15.0, 20.0, "e"),  # start == end (15.0) -> NOT intersecting
        )
        mode, selected = select_segments(
            segments, start_time="00:00:10", end_time="00:00:15"
        )
        self.assertEqual(mode, SelectionMode.TIME_RANGE)
        self.assertEqual([s.index for s in selected], [1, 2, 3])

        # Conflicting selectors rejected
        with self.assertRaises(SelectionError):
            select_segments(segments, segments_spec="1", start_time="00:00:10")

    def test_load_asr_transcript_assigns_stable_0_based_indices(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            asr_path = _write_sample_asr_json(Path(tmp) / "clip.transcript.json")
            transcript = load_asr_transcript(asr_path)
            self.assertEqual(transcript.language, "ko")
            self.assertEqual(len(transcript.segments), 6)
            self.assertEqual([s.index for s in transcript.segments], [0, 1, 2, 3, 4, 5])
            self.assertEqual(transcript.segments[2].start, 10.0)
            self.assertEqual(transcript.segments[2].end, 12.0)
            self.assertEqual(
                transcript.segments[2].text, "아니야 방금 굉장히 인위적이었어"
            )

        with self.assertRaises(TranscriptReadError):
            load_asr_transcript(Path(tmp) / "does_not_exist.transcript.json")

    def test_partial_merge_preserves_other_segments_and_protects_asr_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            asr_path = _write_sample_asr_json(Path(tmp) / "video.transcript.json")
            original_asr_bytes = asr_path.read_bytes()
            out_path = Path(tmp) / "video.zh-TW.translation.json"

            engine = TranslationEngine(
                TranslationConfig(provider="local-llm", batch_size=2),
                backend=_FakeBackend(),
            )

            # First partial run: segments 1-2 (0-based: 0, 1)
            res1 = engine.translate_file(
                TranslationRequest(
                    transcript_path=asr_path,
                    segments_spec="1-2",
                    output_path=out_path,
                )
            )
            self.assertEqual([s.index for s in res1.segments], [0, 1])

            # Second partial run: segment 5 (0-based: 4) with default MERGE
            res2 = engine.translate_file(
                TranslationRequest(
                    transcript_path=asr_path,
                    segments_spec="5",
                    output_path=out_path,
                )
            )
            self.assertEqual([s.index for s in res2.segments], [0, 1, 4])
            loaded = load_translation_json(out_path)
            self.assertEqual([s.index for s in loaded.segments], [0, 1, 4])
            self.assertEqual(loaded.segments[2].source_text, "대박이다")
            self.assertEqual(loaded.segments[2].start, 15.0)
            self.assertEqual(loaded.segments[2].end, 18.0)

            # Guard: attempting to save over the ASR JSON must be rejected!
            with self.assertRaises(TranslationStorageError):
                save_translation_json(
                    res2,
                    asr_path,
                    source_transcript_path=asr_path,
                )

            # Original ASR JSON must be byte-for-byte identical
            self.assertEqual(asr_path.read_bytes(), original_asr_bytes)


class TestCT2NLLBBackend(unittest.TestCase):
    """Tests for CT2-NLLB loading, batch translation, blank index safety, and CUDA fallback."""

    def test_nllb_batch_translation_preserves_indices_with_blank_segments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "nllb"
            model_dir.mkdir()
            (model_dir / "model.bin").write_bytes(b"ct2")
            (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

            fake_translator = _FakeCT2Translator()
            backend = CT2NLLBBackend(
                CT2NLLBConfig(model_path=model_dir, device="auto", compute_type="int8"),
                translator_factory=lambda *a, **kw: fake_translator,
                tokenizer_factory=lambda d: _FakeNLLBTokenizer(),
                cuda_checker=lambda: (True, None),
            )
            backend.load()
            self.assertEqual(backend.device_info, "cuda/int8")

            req = BackendBatchRequest(
                items=(
                    BackendBatchItem(0, "안녕하세요"),
                    BackendBatchItem(1, "   "),
                    BackendBatchItem(2, ""),
                    BackendBatchItem(3, "대박이다"),
                ),
                source_language="ko",
                target_language="zh-TW",
            )
            result = backend.translate_batch(req)
            self.assertEqual(set(result.keys()), {0, 1, 2, 3})
            self.assertEqual(result[0], "譯:안녕하세요")
            self.assertEqual(result[1], "")
            self.assertEqual(result[2], "")
            self.assertEqual(result[3], "譯:대박이다")
            # Only 2 non-empty segments were sent to translate_batch!
            self.assertEqual(len(fake_translator.calls[0]["source"]), 2)
            self.assertEqual(fake_translator.calls[0]["source"][0][0], "kor_Hang")
            self.assertEqual(fake_translator.calls[0]["target_prefix"], [["zho_Hant"], ["zho_Hant"]])

    def test_nllb_degrades_to_cpu_int8_when_cuda_unavailable_or_runtime_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "nllb"
            model_dir.mkdir()
            (model_dir / "model.bin").write_bytes(b"ct2")
            (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

            # Case 1: CUDA unavailable at load time -> degrades to CPU INT8 even with device="cuda" + fallback_to_cpu=True
            recorded_devices: list[tuple[str, str]] = []

            def factory(_dir, *, device, compute_type, cpu_threads=0):
                recorded_devices.append((device, compute_type))
                return _FakeCT2Translator()

            backend1 = CT2NLLBBackend(
                CT2NLLBConfig(model_path=model_dir, device="cuda", compute_type="int8", fallback_to_cpu=True),
                translator_factory=factory,
                tokenizer_factory=lambda d: _FakeNLLBTokenizer(),
                cuda_checker=lambda: (False, "偵測不到可用的 CUDA GPU"),
            )
            backend1.load()
            self.assertEqual(backend1.device_info, "cpu/int8")
            self.assertEqual(recorded_devices, [("cpu", "int8")])
            self.assertIsNotNone(backend1.fallback_notice)

            # Case 2: CUDA succeeds at load, fails with OOM during translate_batch -> auto fallback to CPU INT8
            attempts: list[str] = []

            def runtime_fallback_factory(_dir, *, device, compute_type, cpu_threads=0):
                attempts.append(device)
                if device == "cuda":
                    return _FakeCT2Translator(fail_once_with=RuntimeError("CUDA out of memory"))
                return _FakeCT2Translator()

            backend2 = CT2NLLBBackend(
                CT2NLLBConfig(model_path=model_dir, device="auto", compute_type="int8"),
                translator_factory=runtime_fallback_factory,
                tokenizer_factory=lambda d: _FakeNLLBTokenizer(),
                cuda_checker=lambda: (True, None),
            )
            out = backend2.translate_text("안녕하세요")
            self.assertEqual(out, "譯:안녕하세요")
            self.assertEqual(attempts, ["cuda", "cpu"])
            self.assertEqual(backend2.device_info, "cpu/int8")

    def test_nllb_auto_downloads_when_model_dir_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "models" / "nllb"
            download_calls: list[tuple[str, Path]] = []

            def fake_downloader(repo_id: str, local_dir: Path) -> Path:
                download_calls.append((repo_id, local_dir))
                local_dir.mkdir(parents=True, exist_ok=True)
                (local_dir / "model.bin").write_bytes(b"downloaded")
                (local_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
                return local_dir

            backend = CT2NLLBBackend(
                CT2NLLBConfig(model_path=model_dir, device="cpu"),
                translator_factory=lambda *a, **kw: _FakeCT2Translator(),
                tokenizer_factory=lambda d: _FakeNLLBTokenizer(),
                model_downloader=fake_downloader,
            )
            backend.load()
            self.assertEqual(len(download_calls), 1)
            self.assertEqual(download_calls[0][1], model_dir.resolve())


class TestLocalLLMBackend(unittest.TestCase):
    """Tests for Local LLM OpenAI-compatible requests, alignment parsing, and error handling."""

    def test_local_llm_translates_with_context_and_parses_strict_json(self) -> None:
        captured_urls: list[str] = []
        captured_payloads: list[dict[str, Any]] = []

        def fake_post(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float):
            captured_urls.append(url)
            captured_payloads.append(payload)
            content = json.dumps(
                {
                    "translations": [
                        {"index": 2, "text": "不是啦，剛才實在太刻意了"},
                        {"index": 3, "text": "該不會沒有攝影機吧？"},
                    ]
                },
                ensure_ascii=False,
            )
            if url.endswith("/api/chat"):
                return 200, json.dumps({"message": {"role": "assistant", "content": content}})
            return 200, json.dumps({"choices": [{"message": {"content": content}}]})

        # Ollama with disable_thinking=True uses derived /api/chat with think=False
        backend_ollama = LocalLLMBackend(
            LocalLLMConfig(
                backend="ollama",
                base_url="http://localhost:11434/v1",
                model="qwen3:4b",
                disable_thinking=True,
            ),
            http_post=fake_post,
        )
        req = BackendBatchRequest(
            items=(
                BackendBatchItem(2, "아니야 방금 굉장히 인위적이었어"),
                BackendBatchItem(3, "카메라 없는거 아니야?"),
            ),
            context_before=(BackendBatchItem(1, "됐다 됐다 됐다"),),
            context_after=(BackendBatchItem(4, "어? 그러게 어디있어 카메라?"),),
        )
        result = backend_ollama.translate_batch(req)
        self.assertEqual(
            result,
            {
                2: "不是啦，剛才實在太刻意了",
                3: "該不會沒有攝影機吧？",
            },
        )
        self.assertEqual(captured_urls[0], "http://localhost:11434/api/chat")
        self.assertFalse(captured_payloads[0]["think"])
        user_msg = captured_payloads[0]["messages"][1]["content"]
        self.assertIn("前文參考", user_msg)
        self.assertIn("後文參考", user_msg)

        # llama-server / LM Studio uses OpenAI-compatible /v1/chat/completions
        backend_llama = LocalLLMBackend(
            LocalLLMConfig(
                backend="llama-server",
                base_url="http://localhost:8080/v1",
                model="qwen3-4b-q4",
            ),
            http_post=fake_post,
        )
        result2 = backend_llama.translate_batch(req)
        self.assertEqual(result2, result)
        self.assertEqual(captured_urls[1], "http://localhost:8080/v1/chat/completions")

    def test_alignment_parser_handles_think_blocks_and_fenced_json_and_rejects_bad_ids(self) -> None:
        items = [BackendBatchItem(0, "안녕"), BackendBatchItem(1, "대박")]

        # Valid fenced JSON wrapped after <think> block
        fenced = '<think>thinking...</think>\n```json\n{"translations":[{"index":0,"text":"嗨"},{"index":1,"text":"太扯了"}]}\n```'
        self.assertEqual(
            parse_and_validate_llm_response(fenced, items),
            {0: "嗨", 1: "太扯了"},
        )

        # Singular "translation" wrapper or single-item object still strictly checked
        self.assertEqual(
            parse_and_validate_llm_response(
                '{"translation":[{"index":0,"text":"嗨"},{"index":1,"text":"太扯了"}]}',
                items,
            ),
            {0: "嗨", 1: "太扯了"},
        )
        self.assertEqual(
            parse_and_validate_llm_response(
                '{"index":0,"text":"嗨"}',
                [BackendBatchItem(0, "안녕")],
            ),
            {0: "嗨"},
        )

        # Missing ID -> AlignmentError
        with self.assertRaises(AlignmentError):
            parse_and_validate_llm_response('{"translations":[{"index":0,"text":"嗨"}]}', items)

        # Duplicate ID -> AlignmentError
        with self.assertRaises(AlignmentError):
            parse_and_validate_llm_response(
                '{"translations":[{"index":0,"text":"嗨"},{"index":0,"text":"哈囉"}]}',
                items,
            )

        # Unknown / context ID -> AlignmentError
        with self.assertRaises(AlignmentError):
            parse_and_validate_llm_response(
                '{"translations":[{"index":0,"text":"嗨"},{"index":1,"text":"太扯"},{"index":99,"text":"誤翻"}]}',
                items,
            )

        # Empty translation for non-empty source -> AlignmentError
        with self.assertRaises(AlignmentError):
            parse_and_validate_llm_response(
                '{"translations":[{"index":0,"text":"嗨"},{"index":1,"text":"   "}]}',
                items,
            )

    def test_local_llm_connection_refused_timeout_and_model_not_found_errors(self) -> None:
        # Connection refused -> clear BackendUnavailableError mentioning ct2-nllb fallback
        def conn_refused(*_args):
            raise urllib.error.URLError(ConnectionRefusedError("Connection refused"))

        backend_down = LocalLLMBackend(
            LocalLLMConfig(base_url="http://localhost:11434/v1", model="qwen3:4b", max_retries=0),
            http_post=conn_refused,
        )
        with self.assertRaises(BackendUnavailableError) as ctx:
            backend_down.translate_text("안녕")
        self.assertIn("ct2-nllb", str(ctx.exception))

        # Timeout -> retries and raises BackendTimeoutError
        calls = 0

        def always_timeout(*_args):
            nonlocal calls
            calls += 1
            raise socket.timeout("timed out")

        backend_timeout = LocalLLMBackend(
            LocalLLMConfig(timeout=5, max_retries=1),
            http_post=always_timeout,
            sleep_fn=lambda _: None,
        )
        with self.assertRaises(BackendTimeoutError):
            backend_timeout.translate_text("안녕")
        self.assertEqual(calls, 2)

        # 404 model not found -> clear hint
        backend_404 = LocalLLMBackend(
            LocalLLMConfig(backend="ollama", model="missing:model"),
            http_post=lambda *a: (404, '{"error": "model missing:model not found"}'),
        )
        with self.assertRaises(BackendUnavailableError) as ctx404:
            backend_404.translate_text("안녕")
        self.assertIn("ollama pull missing:model", str(ctx404.exception))


class TestTranslationEngine(unittest.TestCase):
    """Tests for TranslationEngine context windows, shrink-and-retry recovery, cancellation, and APIs."""

    def test_engine_shrinks_batch_on_alignment_error_down_to_single_sentence(self) -> None:
        seen_batch_sizes: list[int] = []

        def flaky_handler(req: BackendBatchRequest) -> dict[int, str]:
            seen_batch_sizes.append(len(req.items))
            # Fail any batch larger than 1 with AlignmentError, succeed on single items
            if len(req.items) > 1:
                raise AlignmentError("malformed batch")
            item = req.items[0]
            return {item.index: f"單句譯:{item.text}"}

        backend = _FakeBackend(handler=flaky_handler)
        engine = TranslationEngine(
            TranslationConfig(provider="local-llm", batch_size=4, context_before=1, context_after=1),
            backend=backend,
        )
        segments = [
            SourceSegment(0, 0.0, 1.0, "첫번째"),
            SourceSegment(1, 1.0, 2.0, "두번째"),
            SourceSegment(2, 2.0, 3.0, "세번째"),
            SourceSegment(3, 3.0, 4.0, "네번째"),
        ]
        results = engine.translate_segments(segments)
        self.assertEqual(len(results), 4)
        self.assertEqual([r.index for r in results], [0, 1, 2, 3])
        self.assertEqual(results[2].translated_text, "單句譯:세번째")
        # 4 -> split to (2, 2) -> each split to (1, 1)
        self.assertEqual(seen_batch_sizes, [4, 2, 1, 1, 2, 1, 1])

    def test_engine_never_writes_corrupt_file_when_alignment_fails_permanently(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            asr_path = _write_sample_asr_json(Path(tmp) / "bad.transcript.json")
            out_path = Path(tmp) / "bad.zh-TW.translation.json"

            backend = _FakeBackend(
                handler=lambda req: (_ for _ in ()).throw(AlignmentError("always broken"))
            )
            engine = TranslationEngine(
                TranslationConfig(provider="local-llm", batch_size=2),
                backend=backend,
                single_item_retries=1,
            )
            with self.assertRaises(AlignmentError):
                engine.translate_file(
                    TranslationRequest(transcript_path=asr_path, output_path=out_path)
                )
            self.assertFalse(out_path.exists())

    def test_engine_supplies_sliding_window_context_even_for_partial_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            asr_path = _write_sample_asr_json(Path(tmp) / "ctx.transcript.json")
            out_path = Path(tmp) / "ctx.zh-TW.translation.json"
            backend = _FakeBackend(supports_context=True)
            engine = TranslationEngine(
                TranslationConfig(
                    provider="local-llm",
                    batch_size=10,
                    context_before=2,
                    context_after=2,
                ),
                backend=backend,
            )
            # Translate only segment 3 (0-based index 2)
            engine.translate_file(
                TranslationRequest(
                    transcript_path=asr_path,
                    segments_spec="3",
                    output_path=out_path,
                )
            )
            self.assertEqual(len(backend.requests), 1)
            req = backend.requests[0]
            self.assertEqual([i.index for i in req.items], [2])
            self.assertEqual([i.index for i in req.context_before], [0, 1])
            self.assertEqual([i.index for i in req.context_after], [3, 4])

    def test_cancellation_stops_before_and_mid_translation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            asr_path = _write_sample_asr_json(Path(tmp) / "cancel.transcript.json")
            out_path = Path(tmp) / "cancel.zh-TW.translation.json"
            token = TranslationCancellationToken()

            def cancel_after_first_batch(req: BackendBatchRequest) -> dict[int, str]:
                token.cancel()
                return {item.index: "ok" for item in req.items}

            backend = _FakeBackend(handler=cancel_after_first_batch)
            engine = TranslationEngine(
                TranslationConfig(provider="local-llm", batch_size=2),
                backend=backend,
            )
            stages: list[TranslationStage] = []
            with self.assertRaises(TranslationCancelledError):
                engine.translate_file(
                    TranslationRequest(transcript_path=asr_path, output_path=out_path),
                    progress_callback=lambda p: stages.append(p.stage),
                    cancel_token=token,
                )
            self.assertIn(TranslationStage.CANCELLED, stages)
            self.assertFalse(out_path.exists())
            # Stopped after batch 1 (6 segments / batch_size 2 = 3 batches total)
            self.assertEqual(len(backend.requests), 1)

    def test_in_memory_apis_and_vram_lifecycle_hooks(self) -> None:
        hook_called = False

        def release_asr_hook() -> None:
            nonlocal hook_called
            hook_called = True

        backend1 = _FakeBackend(name="ct2-nllb", model="nllb", supports_context=False)
        backend2 = _FakeBackend(name="local-llm", model="qwen3:4b", supports_context=True)

        with TranslationEngine(backend=backend1, before_load_hook=release_asr_hook) as engine:
            self.assertEqual(engine.translate_text("안녕하세요"), "繁中(안녕하세요)")
            self.assertTrue(hook_called)
            self.assertTrue(backend1.is_loaded)

            # Switching backend automatically unloads previous backend
            engine.set_backend(backend2)
            self.assertFalse(backend1.is_loaded)
            self.assertEqual(backend1.unload_count, 1)

            segs = engine.translate_segments(["안녕", "잘가"])
            self.assertEqual([s.translated_text for s in segs], ["繁中(안녕)", "繁中(잘가)"])

        # Context manager exit unloads active backend
        self.assertFalse(backend2.is_loaded)
        self.assertEqual(backend2.unload_count, 1)

    def test_skip_existing_policy_only_translates_missing_segments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            asr_path = _write_sample_asr_json(Path(tmp) / "inc.transcript.json")
            out_path = Path(tmp) / "inc.zh-TW.translation.json"
            backend = _FakeBackend()
            engine = TranslationEngine(
                TranslationConfig(provider="local-llm", batch_size=10),
                backend=backend,
            )

            # Translate segments 1-2 first
            engine.translate_file(
                TranslationRequest(
                    transcript_path=asr_path,
                    segments_spec="1-2",
                    output_path=out_path,
                )
            )
            backend.requests.clear()

            # Incremental translate segments 1-4 with SKIP_EXISTING -> only 3-4 sent to backend!
            res = engine.translate_file(
                TranslationRequest(
                    transcript_path=asr_path,
                    segments_spec="1-4",
                    output_path=out_path,
                    existing_policy=ExistingTranslationPolicy.SKIP_EXISTING,
                )
            )
            self.assertEqual(len(backend.requests), 1)
            self.assertEqual([i.index for i in backend.requests[0].items], [2, 3])
            self.assertEqual([s.index for s in res.segments], [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
