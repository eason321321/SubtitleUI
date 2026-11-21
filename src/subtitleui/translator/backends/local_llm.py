"""OpenAI-compatible Local LLM translation backend (Ollama / llama-server / LM Studio)."""

from __future__ import annotations

import json
import re
import socket
import time
import urllib.error
import urllib.request
from typing import Any, Callable

from ..config import CredentialStore, resolve_api_key
from ..errors import (
    AlignmentError,
    BackendError,
    BackendTimeoutError,
    BackendUnavailableError,
)
from ..models import (
    BackendBatchItem,
    BackendBatchRequest,
    CancellationTokenLike,
    LocalLLMConfig,
    check_cancelled,
)
from .base import TranslationBackend

HttpPostFunc = Callable[[str, dict[str, Any], dict[str, str], float], tuple[int, str]]

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", flags=re.DOTALL | re.IGNORECASE)

_LANGUAGE_DISPLAY_NAMES: dict[str, str] = {
    "ko": "韓文",
    "ko-kr": "韓文",
    "zh-tw": "台灣繁體中文",
    "zh-hant": "繁體中文",
    "zh-cn": "簡體中文",
    "en": "英文",
    "ja": "日文",
}

TRANSLATION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "translations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "text": {"type": "string"},
                },
                "required": ["index", "text"],
            },
        }
    },
    "required": ["translations"],
}


def _display_language(code: str) -> str:
    return _LANGUAGE_DISPLAY_NAMES.get(code.strip().lower(), code)


def _strip_thinking_blocks(text: str) -> str:
    cleaned = _THINK_BLOCK_RE.sub("", text)
    if "</think>" in cleaned.lower():
        idx = cleaned.lower().rfind("</think>")
        if idx != -1:
            cleaned = cleaned[idx + len("</think>") :]
    return cleaned.strip()


def _extract_json_candidates(raw_text: str) -> list[str]:
    """Return ordered JSON string candidates for direct parse and re-parse recovery."""
    cleaned = _strip_thinking_blocks(raw_text)
    candidates: list[str] = []
    if cleaned:
        candidates.append(cleaned)

    for match in _CODE_FENCE_RE.finditer(cleaned):
        fenced = match.group(1).strip()
        if fenced and fenced not in candidates:
            candidates.append(fenced)

    first_obj = cleaned.find("{")
    last_obj = cleaned.rfind("}")
    if first_obj != -1 and last_obj > first_obj:
        snippet = cleaned[first_obj : last_obj + 1].strip()
        if snippet not in candidates:
            candidates.append(snippet)

    first_arr = cleaned.find("[")
    last_arr = cleaned.rfind("]")
    if first_arr != -1 and last_arr > first_arr:
        snippet = cleaned[first_arr : last_arr + 1].strip()
        if snippet not in candidates:
            candidates.append(snippet)

    return candidates


def parse_and_validate_llm_response(
    raw_content: str,
    expected_items: list[BackendBatchItem],
) -> dict[int, str]:
    """Parse an LLM response and strictly verify 1-to-1 segment index alignment.

    Checks performed:
    - Valid JSON structure (with fenced/substring re-parse fallback)
    - No duplicate indices
    - No unknown/out-of-batch indices (e.g. context lines mistakenly returned)
    - No missing indices (no dropped segments)
    - Every translation is a valid string (and non-empty for non-empty source text)

    Raises:
        AlignmentError: If any parsing or alignment check fails.
    """
    if not isinstance(raw_content, str) or not raw_content.strip():
        raise AlignmentError("LLM 回傳內容為空白，無法解析翻譯結果。")

    parsed: Any = None
    last_json_err: Exception | None = None
    for candidate in _extract_json_candidates(raw_content):
        try:
            parsed = json.loads(candidate)
            break
        except json.JSONDecodeError as exc:
            last_json_err = exc

    if parsed is None:
        raise AlignmentError(f"LLM 回傳內容無法解析為合法 JSON：{last_json_err}")

    expected_map = {item.index: item.text.strip() for item in expected_items}
    expected_ids = set(expected_map.keys())
    extracted: dict[int, str] = {}

    if isinstance(parsed, dict) and ("translations" in parsed or "translation" in parsed):
        wrapper_key = "translations" if "translations" in parsed else "translation"
        entries = parsed[wrapper_key]
        if isinstance(entries, list):
            _collect_list_entries(entries, extracted)
        elif isinstance(entries, dict):
            _collect_list_entries([entries], extracted)
        elif (
            isinstance(entries, str)
            and len(expected_items) == 1
            and "index" in parsed
        ):
            _collect_list_entries(
                [{"index": parsed["index"], "text": entries}], extracted
            )
        else:
            raise AlignmentError(
                f"LLM 回傳 JSON 的 `{wrapper_key}` 欄位格式不符預期。"
            )
    elif isinstance(parsed, dict) and ("index" in parsed or "id" in parsed):
        _collect_list_entries([parsed], extracted)
    elif isinstance(parsed, list):
        _collect_list_entries(parsed, extracted)
    elif isinstance(parsed, dict):
        if not parsed:
            raise AlignmentError("LLM 回傳空的 JSON 物件。")
        for key, value in parsed.items():
            try:
                idx = int(str(key).strip())
            except ValueError as exc:
                raise AlignmentError(
                    f"LLM 回傳 JSON 鍵名「{key}」不是有效的句子索引整數。"
                ) from exc
            if idx in extracted:
                raise AlignmentError(f"LLM 回傳包含重複的句子索引：{idx}")
            if not isinstance(value, str):
                raise AlignmentError(
                    f"句子索引 {idx} 的翻譯結果必須是字串，目前為 {type(value).__name__}。"
                )
            extracted[idx] = value.strip()
    else:
        raise AlignmentError(
            f"LLM 回傳的 JSON 頂層型別不符預期（收到 {type(parsed).__name__}）。"
        )

    returned_ids = set(extracted.keys())
    unknown_ids = sorted(returned_ids - expected_ids)
    if unknown_ids:
        raise AlignmentError(
            f"LLM 回傳包含不存在或屬於上下文的句子索引：{unknown_ids}（預期：{sorted(expected_ids)}）。"
        )

    missing_ids = sorted(expected_ids - returned_ids)
    if missing_ids:
        raise AlignmentError(
            f"LLM 漏翻部分句子，缺少索引：{missing_ids}（預期：{sorted(expected_ids)}）。"
        )

    empty_ids = [
        idx
        for idx, src_text in expected_map.items()
        if src_text and not extracted[idx].strip()
    ]
    if empty_ids:
        raise AlignmentError(f"LLM 對非空白原文回傳了空白翻譯，索引：{empty_ids}。")

    return extracted


def _collect_list_entries(entries: list[Any], target: dict[int, str]) -> None:
    for pos, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise AlignmentError(
                f"LLM 回傳的第 {pos + 1} 筆項目不是物件（收到 {type(entry).__name__}）。"
            )
        raw_idx = entry.get("index", entry.get("id"))
        if raw_idx is None or isinstance(raw_idx, bool):
            raise AlignmentError(f"LLM 回傳的第 {pos + 1} 筆項目缺少有效的 `index` 欄位。")
        try:
            idx = int(raw_idx)
        except (TypeError, ValueError) as exc:
            raise AlignmentError(
                f"LLM 回傳的第 {pos + 1} 筆項目 `index` 非整數：{raw_idx!r}"
            ) from exc

        if idx in target:
            raise AlignmentError(f"LLM 回傳包含重複的句子索引：{idx}")

        raw_text = entry.get(
            "text",
            entry.get("translated_text", entry.get("translation")),
        )
        if not isinstance(raw_text, str):
            raise AlignmentError(
                f"句子索引 {idx} 的翻譯欄位必須是字串，目前為 {type(raw_text).__name__}。"
            )
        target[idx] = raw_text.strip()


def build_translation_messages(
    request: BackendBatchRequest,
    active_items: list[BackendBatchItem],
) -> list[dict[str, str]]:
    """Build system and user messages for context-aware Taiwanese Traditional Chinese translation."""
    src_label = _display_language(request.source_language)
    tgt_label = _display_language(request.target_language)

    system_lines = [
        f"你是一位專業的影視字幕翻譯人員，負責將{src_label}口語逐字稿精準翻譯為自然道地的{tgt_label}（{request.target_language}）。",
        "請嚴格遵守以下翻譯原則：",
        "1. 使用自然、符合台灣日常口語習慣的繁體中文（避免中國大陸用語或簡體字）。",
        "2. 忠實保留原文意思，不可擅自補充原文沒有的資訊，也不可為了流暢而扭曲原意。",
        "3. 正確處理韓文的半語（반말）、敬語（존댓말）、口語縮略、俚語與感嘆詞，保留原句語氣、情緒與疑問句感。",
        "4. 參考「前文參考」與「後文參考」判斷省略的主詞、代名詞與語境，但絕對不要翻譯參考句，也不要把參考句的編號放進輸出。",
        "5. 只翻譯「待翻譯句子」中的每一句，嚴格保留原本的整數 index，不可漏翻、重複、合併、拆句或更改 index。",
        '6. 只回傳嚴格的 JSON 物件，不要輸出任何解釋文字或 Markdown 標記。輸出範例：{"translations": [{"index": 10, "text": "不是啦，剛才實在太刻意了"}]}',
    ]

    user_sections: list[str] = []
    if request.context_before:
        before_lines = "\n".join(
            f"({item.index}) {item.text}" for item in request.context_before
        )
        user_sections.append(f"[前文參考（僅供理解語境，請勿翻譯）]\n{before_lines}")

    target_lines = "\n".join(f"[{item.index}] {item.text}" for item in active_items)
    user_sections.append(f"[待翻譯{src_label}句子]\n{target_lines}")

    if request.context_after:
        after_lines = "\n".join(
            f"({item.index}) {item.text}" for item in request.context_after
        )
        user_sections.append(f"[後文參考（僅供理解語境，請勿翻譯）]\n{after_lines}")

    expected_indices = ", ".join(str(item.index) for item in active_items)
    user_sections.append(
        f"請將上述 [待翻譯{src_label}句子]（共 {len(active_items)} 句，index 為 {expected_indices}）"
        f"全部翻譯成{tgt_label}，並只回傳 "
        '{"translations": [{"index": <整數>, "text": "<翻譯文字>"}]} JSON。'
    )

    return [
        {"role": "system", "content": "\n".join(system_lines)},
        {"role": "user", "content": "\n\n".join(user_sections)},
    ]


def _default_http_post(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
) -> tuple[int, str]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = getattr(resp, "status", 200)
            raw = resp.read().decode("utf-8", errors="replace")
            return int(status), raw
    except urllib.error.HTTPError as exc:
        err_body = ""
        try:
            err_body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        return int(exc.code), err_body


class LocalLLMBackend(TranslationBackend):
    """OpenAI-compatible HTTP client backend for Ollama, llama-server, and LM Studio."""

    def __init__(
        self,
        config: LocalLLMConfig | None = None,
        *,
        http_post: HttpPostFunc | None = None,
        credential_store: CredentialStore | None = None,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> None:
        self._config = config or LocalLLMConfig()
        self._http_post = http_post or _default_http_post
        self._credential_store = credential_store
        self._sleep = sleep_fn or time.sleep
        self._loaded = False

    @property
    def name(self) -> str:
        return "local-llm"

    @property
    def model_name(self) -> str:
        return self._config.model

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def device_info(self) -> str | None:
        return f"api/{self._config.backend}"

    @property
    def supports_context(self) -> bool:
        return True

    @property
    def _use_ollama_no_think_chat(self) -> bool:
        """Use Ollama's native /api/chat endpoint when disable_thinking is enabled.

        Ollama's OpenAI `/v1/chat/completions` layer ignores `"think": false` on
        reasoning models like Qwen3, whereas `/api/chat` honors `"think": false`
        and structured JSON schema output natively (avoiding 30x slowdown).
        All URLs are still derived dynamically from the configured `base_url`.
        """
        return (
            self._config.backend.strip().lower() == "ollama"
            and self._config.disable_thinking
            and not self._config.base_url.rstrip("/").endswith("/chat/completions")
        )

    @property
    def endpoint_url(self) -> str:
        base = self._config.base_url.rstrip("/")
        if self._use_ollama_no_think_chat:
            if base.endswith("/v1"):
                return f"{base[:-3]}/api/chat"
            if base.endswith("/api/chat"):
                return base
            return f"{base}/api/chat"
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"

    def load(self) -> None:
        if not self._config.base_url.strip():
            raise BackendUnavailableError("Local LLM 的 `base_url` 未設定。")
        if not self._config.model.strip():
            raise BackendUnavailableError("Local LLM 的 `model` 名稱未設定。")
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def translate_batch(
        self,
        request: BackendBatchRequest,
        *,
        cancel_token: CancellationTokenLike | None = None,
    ) -> dict[int, str]:
        check_cancelled(cancel_token)
        if not request.items:
            return {}
        if not self._loaded:
            self.load()

        results: dict[int, str] = {}
        active_items: list[BackendBatchItem] = []
        for item in request.items:
            if not item.text.strip():
                results[item.index] = ""
            else:
                active_items.append(item)

        if not active_items:
            return results

        messages = build_translation_messages(request, active_items)
        if self._use_ollama_no_think_chat:
            payload: dict[str, Any] = {
                "model": self._config.model,
                "messages": messages,
                "think": False,
                "format": TRANSLATION_JSON_SCHEMA,
                "stream": False,
                "options": {"temperature": self._config.temperature},
            }
        else:
            payload = {
                "model": self._config.model,
                "messages": messages,
                "temperature": self._config.temperature,
                "stream": False,
            }

        headers = {"Content-Type": "application/json"}
        api_key = resolve_api_key(
            self._config, credential_store=self._credential_store
        )
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        raw_content = self._post_with_retry(
            payload=payload,
            headers=headers,
            cancel_token=cancel_token,
        )
        parsed = parse_and_validate_llm_response(raw_content, active_items)
        results.update(parsed)
        return results

    def _post_with_retry(
        self,
        *,
        payload: dict[str, Any],
        headers: dict[str, str],
        cancel_token: CancellationTokenLike | None,
    ) -> str:
        max_attempts = max(1, int(self._config.max_retries) + 1)
        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            check_cancelled(cancel_token)
            try:
                status, response_body = self._http_post(
                    self.endpoint_url,
                    payload,
                    headers,
                    self._config.timeout,
                )
            except (socket.timeout, TimeoutError) as exc:
                last_error = BackendTimeoutError(
                    f"呼叫 Local LLM API（{self.endpoint_url}，模型：{self._config.model}）逾時（>{self._config.timeout}s）。"
                    "若硬體 VRAM 有限或啟用 CPU Offload，建議增加 config.yaml 的 timeout 或縮小 batch_size。"
                )
                if attempt < max_attempts:
                    self._sleep(min(2.0 * attempt, 5.0))
                    continue
                raise last_error from exc
            except urllib.error.URLError as exc:
                reason = exc.reason
                if isinstance(reason, (socket.timeout, TimeoutError)):
                    last_error = BackendTimeoutError(
                        f"呼叫 Local LLM API（{self.endpoint_url}）逾時（>{self._config.timeout}s）。"
                    )
                    if attempt < max_attempts:
                        self._sleep(min(2.0 * attempt, 5.0))
                        continue
                    raise last_error from exc
                raise BackendUnavailableError(
                    f"無法連線至 Local LLM 服務（{self._config.backend} @ {self._config.base_url}）：{reason}。"
                    "請確認本機 LLM 服務已啟動、base_url 設定正確，或改用 `--backend ct2-nllb` 離線翻譯。"
                ) from exc
            except OSError as exc:
                raise BackendUnavailableError(
                    f"無法連線至 Local LLM 服務（{self._config.base_url}）：{exc}。"
                    "請確認本機 LLM 服務已啟動，或改用 `--backend ct2-nllb`。"
                ) from exc

            if status == 200:
                return self._extract_message_content(response_body)

            if status in (401, 403):
                raise BackendUnavailableError(
                    f"Local LLM API 驗證失敗（HTTP {status}）。請檢查 API Key 或環境變數設定。"
                )

            if status == 404 or "not found" in response_body.lower():
                hint = (
                    f"若使用 Ollama，請先執行 `ollama pull {self._config.model}`。"
                    if self._config.backend == "ollama"
                    else "請確認伺服器已載入指定模型且 base_url 設定正確。"
                )
                raise BackendUnavailableError(
                    f"Local LLM 找不到指定模型「{self._config.model}」或端點（HTTP {status}：{self._summarize_body(response_body)}）。{hint}"
                )

            if status in (429, 500, 502, 503, 504):
                last_error = BackendError(
                    f"Local LLM 服務暫時性錯誤（HTTP {status}）：{self._summarize_body(response_body)}"
                )
                if attempt < max_attempts:
                    self._sleep(min(2.0 * attempt, 5.0))
                    continue
                raise last_error

            raise BackendError(
                f"Local LLM API 回傳錯誤（HTTP {status}）：{self._summarize_body(response_body)}"
            )

        assert last_error is not None
        raise last_error

    @staticmethod
    def _summarize_body(body: str, limit: int = 200) -> str:
        compact = " ".join(body.split())
        if len(compact) <= limit:
            return compact or "(空回應)"
        return compact[:limit] + "..."

    @staticmethod
    def _extract_message_content(response_body: str) -> str:
        try:
            data = json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise AlignmentError(
                f"Local LLM API 回傳非 JSON HTTP 本文：{exc}"
            ) from exc

        if not isinstance(data, dict):
            raise AlignmentError("Local LLM API 回傳格式無效（頂層非物件）。")

        # OpenAI-compatible `/v1/chat/completions` format
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict) and isinstance(message.get("content"), str):
                    return message["content"]
                if isinstance(first.get("text"), str):
                    return first["text"]

        # Ollama native `/api/chat` format
        direct_msg = data.get("message")
        if isinstance(direct_msg, dict) and isinstance(direct_msg.get("content"), str):
            return direct_msg["content"]

        raise AlignmentError(
            "Local LLM API 回傳缺少有效的回應文字（choices[0].message.content 或 message.content）。"
        )
