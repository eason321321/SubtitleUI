"""CTranslate2 NLLB-200-distilled-600M local translation backend (PyTorch-free)."""

from __future__ import annotations

import ctypes
import gc
import os
import site
import sys
from pathlib import Path
from typing import Any, Callable, Protocol

from ..errors import BackendError, BackendUnavailableError
from ..models import (
    BackendBatchRequest,
    CT2NLLBConfig,
    CancellationTokenLike,
    check_cancelled,
    resolve_nllb_language_code,
)
from .base import TranslationBackend

_REGISTERED_DLL_HANDLES: list[Any] = []
_REGISTERED_DLL_DIRS: set[Path] = set()

_WINDOWS_CUDA_DLLS = ("cublas64_12.dll",)

_CUDA_ERROR_KEYWORDS = (
    "cuda",
    "cublas",
    "cudnn",
    "out of memory",
    "cannot be loaded",
    "could not locate",
    "unsupported device",
    "no cuda",
    "driver",
)

_NLLB_REQUIRED_MODEL_FILE = "model.bin"
_NLLB_TOKENIZER_FILES = ("tokenizer.json", "sentencepiece.bpe.model")


class NLLBTokenizerLike(Protocol):
    """Minimal tokenizer interface required by CT2NLLBBackend."""

    def encode(self, text: str, src_lang: str) -> list[str]:
        ...

    def decode(self, tokens: list[str], tgt_lang: str) -> str:
        ...


class _HFTokenizersNLLBWrapper:
    """NLLB tokenizer using HuggingFace `tokenizers` (`tokenizer.json`, zero PyTorch)."""

    def __init__(self, tokenizer_json_path: Path) -> None:
        from tokenizers import Tokenizer

        self._tokenizer = Tokenizer.from_file(str(tokenizer_json_path))

    def encode(self, text: str, src_lang: str) -> list[str]:
        # Encode raw text pieces without default post-processor, then explicitly
        # wrap with NLLB's required `[src_lang, *pieces, "</s>"]` sequence.
        encoding = self._tokenizer.encode(text, add_special_tokens=False)
        tokens = [
            tok for tok in encoding.tokens if tok not in ("<s>", "</s>", "<pad>")
        ]
        return [src_lang, *tokens, "</s>"]

    def decode(self, tokens: list[str], tgt_lang: str) -> str:
        filtered = [
            tok
            for tok in tokens
            if tok not in (tgt_lang, "<s>", "</s>", "<pad>")
        ]
        if not filtered:
            return ""
        ids: list[int] = []
        for tok in filtered:
            tok_id = self._tokenizer.token_to_id(tok)
            if tok_id is not None:
                ids.append(tok_id)
        if ids:
            decoded = self._tokenizer.decode(ids, skip_special_tokens=True)
            return decoded.strip()
        # Fallback manual SentencePiece join if token lookup misses any piece.
        return "".join(filtered).replace("▁", " ").strip()


class _SentencePieceNLLBWrapper:
    """NLLB tokenizer using `sentencepiece` (`sentencepiece.bpe.model`, zero PyTorch)."""

    def __init__(self, spm_model_path: Path) -> None:
        import sentencepiece as spm

        processor = spm.SentencePieceProcessor()
        processor.load(str(spm_model_path))
        self._sp = processor

    def encode(self, text: str, src_lang: str) -> list[str]:
        pieces = list(self._sp.encode(text, out_type=str))
        return [src_lang, *pieces, "</s>"]

    def decode(self, tokens: list[str], tgt_lang: str) -> str:
        filtered = [
            tok
            for tok in tokens
            if tok not in (tgt_lang, "<s>", "</s>", "<pad>")
        ]
        if not filtered:
            return ""
        return str(self._sp.decode(filtered)).strip()


def _register_nvidia_dll_directories() -> list[Path]:
    """Register pip-installed NVIDIA runtime DLL directories on Windows."""
    if sys.platform != "win32":
        return []

    candidate_roots: list[Path] = []
    try:
        for sp in site.getsitepackages():
            candidate_roots.append(Path(sp) / "nvidia")
    except Exception:
        pass
    candidate_roots.append(Path(sys.prefix) / "Lib" / "site-packages" / "nvidia")

    added: list[Path] = []
    current_path = os.environ.get("PATH", "")
    path_entries = [entry for entry in current_path.split(os.pathsep) if entry]
    path_lower = {entry.lower() for entry in path_entries}

    for root in candidate_roots:
        if not root.is_dir():
            continue
        for bin_dir in sorted(root.glob("*/bin")):
            if not bin_dir.is_dir():
                continue
            resolved = bin_dir.resolve()
            if resolved in _REGISTERED_DLL_DIRS:
                continue
            try:
                if hasattr(os, "add_dll_directory"):
                    handle = os.add_dll_directory(str(resolved))
                    _REGISTERED_DLL_HANDLES.append(handle)
                if str(resolved).lower() not in path_lower:
                    path_entries.insert(0, str(resolved))
                    path_lower.add(str(resolved).lower())
                _REGISTERED_DLL_DIRS.add(resolved)
                added.append(resolved)
            except OSError:
                continue

    if added:
        os.environ["PATH"] = os.pathsep.join(path_entries)

    return added


def check_ct2_cuda_availability(
    *,
    cuda_device_count_fn: Callable[[], int] | None = None,
    dll_check_fn: Callable[[str], bool] | None = None,
    platform_name: str | None = None,
) -> tuple[bool, str | None]:
    """Check whether CUDA is usable for CTranslate2 NLLB translation."""
    active_platform = platform_name or sys.platform
    if active_platform == "win32":
        _register_nvidia_dll_directories()

    if cuda_device_count_fn is None:
        def _count() -> int:
            import ctranslate2

            return int(ctranslate2.get_cuda_device_count())

        count_fn = _count
    else:
        count_fn = cuda_device_count_fn

    try:
        device_count = count_fn()
    except Exception as exc:
        return False, f"無法偵測 CUDA 裝置（{exc}）"

    if device_count <= 0:
        return False, "偵測不到可用的 CUDA GPU"

    if active_platform == "win32":
        if dll_check_fn is None:
            def _check_dll(dll_name: str) -> bool:
                try:
                    ctypes.WinDLL(dll_name)
                    return True
                except OSError:
                    return False

            checker = _check_dll
        else:
            checker = dll_check_fn

        missing = [dll for dll in _WINDOWS_CUDA_DLLS if not checker(dll)]
        if missing:
            return False, f"缺少 CUDA 執行期函式庫（{', '.join(missing)}）"

    return True, None


def _is_cuda_runtime_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(keyword in text for keyword in _CUDA_ERROR_KEYWORDS)


def _has_complete_local_model(model_dir: Path) -> bool:
    if not model_dir.is_dir():
        return False
    if not (model_dir / _NLLB_REQUIRED_MODEL_FILE).is_file():
        return False
    return any((model_dir / fname).is_file() for fname in _NLLB_TOKENIZER_FILES)


def _default_model_downloader(repo_id: str, local_dir: Path) -> Path:
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise BackendUnavailableError(
            "尚未安裝 huggingface_hub，無法自動下載 CT2-NLLB 模型。"
        ) from exc

    local_dir.mkdir(parents=True, exist_ok=True)
    try:
        downloaded = snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            allow_patterns=[
                "model.bin",
                "config.json",
                "shared_vocabulary.*",
                "source_vocabulary.*",
                "target_vocabulary.*",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "sentencepiece.bpe.model",
            ],
        )
    except Exception as exc:
        raise BackendUnavailableError(
            f"無法從 Hugging Face 下載 CT2-NLLB 模型「{repo_id}」至「{local_dir}」：{exc}"
        ) from exc

    return Path(downloaded).resolve()


def _default_tokenizer_factory(model_dir: Path) -> NLLBTokenizerLike:
    spm_path = model_dir / "sentencepiece.bpe.model"
    tok_json_path = model_dir / "tokenizer.json"

    # Prefer SentencePiece if installed and model file exists; otherwise reuse
    # already-installed HuggingFace `tokenizers` (`tokenizer.json`, zero extra deps).
    if spm_path.is_file():
        try:
            return _SentencePieceNLLBWrapper(spm_path)
        except ImportError:
            pass

    if tok_json_path.is_file():
        try:
            return _HFTokenizersNLLBWrapper(tok_json_path)
        except ImportError as exc:
            raise BackendUnavailableError(
                "缺少 tokenizer 套件（tokenizers 或 sentencepiece），無法載入 NLLB 分詞器。"
            ) from exc

    raise BackendUnavailableError(
        f"模型資料夾「{model_dir}」缺少分詞器檔案（tokenizer.json 或 sentencepiece.bpe.model）。"
    )


def _default_translator_factory(
    model_dir: Path,
    *,
    device: str,
    compute_type: str,
    cpu_threads: int = 0,
) -> Any:
    _register_nvidia_dll_directories()
    try:
        import ctranslate2
    except ImportError as exc:
        raise BackendUnavailableError(
            "尚未安裝 ctranslate2，無法使用 CT2-NLLB 翻譯後端。"
        ) from exc

    kwargs: dict[str, Any] = {
        "device": device,
        "compute_type": compute_type,
    }
    if device == "cpu" and cpu_threads > 0:
        kwargs["intra_threads"] = cpu_threads
    return ctranslate2.Translator(str(model_dir), **kwargs)


class CT2NLLBBackend(TranslationBackend):
    """Local PyTorch-free CTranslate2 NLLB-200-distilled-600M translation backend."""

    def __init__(
        self,
        config: CT2NLLBConfig | None = None,
        *,
        translator_factory: Callable[..., Any] | None = None,
        tokenizer_factory: Callable[[Path], NLLBTokenizerLike] | None = None,
        model_downloader: Callable[[str, Path], Path] | None = None,
        cuda_checker: Callable[[], tuple[bool, str | None]] | None = None,
    ) -> None:
        self._config = config or CT2NLLBConfig()
        self._translator_factory = translator_factory or _default_translator_factory
        self._tokenizer_factory = tokenizer_factory or _default_tokenizer_factory
        self._model_downloader = model_downloader or _default_model_downloader
        self._cuda_checker = cuda_checker or check_ct2_cuda_availability

        self._translator: Any | None = None
        self._tokenizer: NLLBTokenizerLike | None = None
        self._active_device: str | None = None
        self._active_compute_type: str | None = None
        self._fallback_notice: str | None = None

    @property
    def name(self) -> str:
        return "ct2-nllb"

    @property
    def model_name(self) -> str:
        return self._config.model_repo

    @property
    def is_loaded(self) -> bool:
        return self._translator is not None and self._tokenizer is not None

    @property
    def device_info(self) -> str | None:
        if self._active_device and self._active_compute_type:
            return f"{self._active_device}/{self._active_compute_type}"
        return None

    @property
    def supports_context(self) -> bool:
        return False

    @property
    def fallback_notice(self) -> str | None:
        return self._fallback_notice

    def _ensure_model_dir(self) -> Path:
        model_dir = Path(self._config.model_path).expanduser().resolve()
        if _has_complete_local_model(model_dir):
            return model_dir

        downloaded = self._model_downloader(self._config.model_repo, model_dir)
        if not _has_complete_local_model(downloaded):
            raise BackendUnavailableError(
                f"CT2-NLLB 模型下載完成但缺少必要檔案（model.bin 或 tokenizer）：{downloaded}"
            )
        return downloaded

    def _resolve_device_candidates(self) -> list[tuple[str, str, str | None]]:
        preferred = self._config.device.strip().lower()
        compute_type = self._config.compute_type.strip() or "int8"

        if preferred == "cpu":
            return [("cpu", compute_type, None)]

        cuda_ok, cuda_reason = self._cuda_checker()

        if preferred == "cuda":
            if cuda_ok:
                candidates: list[tuple[str, str, str | None]] = [
                    ("cuda", compute_type, None)
                ]
                if self._config.fallback_to_cpu:
                    candidates.append(
                        (
                            "cpu",
                            "int8",
                            "CUDA 載入 CT2-NLLB 失敗，已自動降級為 CPU (INT8)",
                        )
                    )
                return candidates

            if self._config.fallback_to_cpu:
                reason = (
                    f"{cuda_reason}，已自動降級為 CPU (INT8)"
                    if cuda_reason
                    else "CUDA 不可用，已自動降級為 CPU (INT8)"
                )
                return [("cpu", "int8", reason)]

            raise BackendUnavailableError(
                f"已指定 CT2-NLLB 使用 CUDA，但目前無法啟用 GPU：{cuda_reason or '未知原因'}。"
                "請將設定檔 device 改為 auto / cpu，或啟用 fallback_to_cpu。"
            )

        # preferred == "auto"
        if cuda_ok:
            return [
                ("cuda", compute_type, None),
                ("cpu", "int8", "CUDA 載入 CT2-NLLB 失敗，已自動降級為 CPU (INT8)"),
            ]

        reason = (
            f"{cuda_reason}，已自動降級為 CPU (INT8)"
            if cuda_reason
            else "CUDA 不可用，已自動降級為 CPU (INT8)"
        )
        return [("cpu", "int8", reason)]

    def load(self) -> None:
        if self.is_loaded:
            return

        model_dir = self._ensure_model_dir()
        tokenizer = self._tokenizer_factory(model_dir)
        candidates = self._resolve_device_candidates()
        last_error: Exception | None = None

        for idx, (device, compute_type, notice) in enumerate(candidates):
            try:
                translator = self._translator_factory(
                    model_dir,
                    device=device,
                    compute_type=compute_type,
                    cpu_threads=self._config.cpu_threads,
                )
            except Exception as exc:
                last_error = exc
                has_fallback = idx + 1 < len(candidates)
                if has_fallback and device == "cuda":
                    continue
                raise BackendUnavailableError(
                    f"載入 CT2-NLLB 模型失敗（{device}/{compute_type}）：{exc}"
                ) from exc

            self._translator = translator
            self._tokenizer = tokenizer
            self._active_device = device
            self._active_compute_type = compute_type
            self._fallback_notice = notice
            return

        raise BackendUnavailableError(f"無法載入 CT2-NLLB 模型：{last_error}")

    def unload(self) -> None:
        self._translator = None
        self._tokenizer = None
        self._active_device = None
        self._active_compute_type = None
        gc.collect()

    def translate_batch(
        self,
        request: BackendBatchRequest,
        *,
        cancel_token: CancellationTokenLike | None = None,
    ) -> dict[int, str]:
        check_cancelled(cancel_token)

        if not request.items:
            return {}

        if not self.is_loaded:
            self.load()

        try:
            src_code = resolve_nllb_language_code(request.source_language)
            tgt_code = resolve_nllb_language_code(request.target_language)
        except ValueError as exc:
            raise BackendError(str(exc)) from exc

        results: dict[int, str] = {}
        non_empty_indices: list[int] = []
        non_empty_texts: list[str] = []

        # Preserve exact index mapping even when segments are blank or whitespace-only.
        for item in request.items:
            cleaned = item.text.strip()
            if not cleaned:
                results[item.index] = ""
            else:
                non_empty_indices.append(item.index)
                non_empty_texts.append(cleaned)

        if not non_empty_texts:
            return results

        check_cancelled(cancel_token)
        assert self._tokenizer is not None
        assert self._translator is not None

        tokenized_sources = [
            self._tokenizer.encode(text, src_code) for text in non_empty_texts
        ]
        target_prefixes = [[tgt_code] for _ in non_empty_texts]

        raw_outputs = self._run_translate_batch_with_fallback(
            tokenized_sources=tokenized_sources,
            target_prefixes=target_prefixes,
            cancel_token=cancel_token,
        )

        if len(raw_outputs) != len(non_empty_indices):
            raise BackendError(
                f"CT2-NLLB 批次翻譯回傳數量不符：預期 {len(non_empty_indices)} 筆，實際 {len(raw_outputs)} 筆。"
            )

        for idx, output in zip(non_empty_indices, raw_outputs):
            hypotheses = getattr(output, "hypotheses", None)
            if not hypotheses:
                results[idx] = ""
                continue
            results[idx] = self._tokenizer.decode(list(hypotheses[0]), tgt_code)

        return results

    def _run_translate_batch_with_fallback(
        self,
        *,
        tokenized_sources: list[list[str]],
        target_prefixes: list[list[str]],
        cancel_token: CancellationTokenLike | None,
    ) -> list[Any]:
        assert self._translator is not None
        check_cancelled(cancel_token)

        try:
            return list(
                self._translator.translate_batch(
                    tokenized_sources,
                    target_prefix=target_prefixes,
                    beam_size=self._config.beam_size,
                    max_decoding_length=self._config.max_decoding_length,
                )
            )
        except Exception as exc:
            can_fallback = (
                self._active_device == "cuda"
                and (
                    self._config.device.strip().lower() == "auto"
                    or self._config.fallback_to_cpu
                )
                and _is_cuda_runtime_error(exc)
            )
            if not can_fallback:
                raise BackendError(f"CT2-NLLB 翻譯失敗：{exc}") from exc

            # Fall back to CPU INT8 cleanly
            model_dir = Path(self._config.model_path).expanduser().resolve()
            self._translator = None
            gc.collect()
            try:
                self._translator = self._translator_factory(
                    model_dir,
                    device="cpu",
                    compute_type="int8",
                    cpu_threads=self._config.cpu_threads,
                )
            except Exception as fallback_exc:
                raise BackendUnavailableError(
                    f"CUDA 推論失敗且切換 CPU INT8 亦失敗：{fallback_exc}"
                ) from fallback_exc

            self._active_device = "cpu"
            self._active_compute_type = "int8"
            self._fallback_notice = (
                f"CUDA 推論失敗（{exc}），已自動降級為 CPU (INT8)"
            )
            check_cancelled(cancel_token)
            try:
                return list(
                    self._translator.translate_batch(
                        tokenized_sources,
                        target_prefix=target_prefixes,
                        beam_size=self._config.beam_size,
                        max_decoding_length=self._config.max_decoding_length,
                    )
                )
            except Exception as retry_exc:
                raise BackendError(
                    f"CT2-NLLB 切換至 CPU INT8 後翻譯仍失敗：{retry_exc}"
                ) from retry_exc
