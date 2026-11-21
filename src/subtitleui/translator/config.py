"""Configuration manager for SubtitleUI Phase 3 translation settings."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Protocol

import yaml

from ..paths import DEFAULT_CONFIG_PATH, PROJECT_ROOT
from .errors import ConfigError
from .models import (
    CT2NLLBConfig,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CONTEXT_AFTER,
    DEFAULT_CONTEXT_BEFORE,
    DEFAULT_LOCAL_LLM_BACKEND,
    DEFAULT_LOCAL_LLM_BASE_URL,
    DEFAULT_LOCAL_LLM_MODEL,
    DEFAULT_LOCAL_LLM_TIMEOUT,
    DEFAULT_NLLB_MODEL_REPO,
    DEFAULT_PROVIDER,
    DEFAULT_SOURCE_LANGUAGE,
    DEFAULT_TARGET_LANGUAGE,
    LocalLLMConfig,
    SUPPORTED_NLLB_DEVICES,
    SUPPORTED_PROVIDERS,
    TranslationConfig,
)

DEFAULT_CONFIG_YAML = """# SubtitleUI 翻譯系統設定檔
# 修改後重新執行程式即可生效；若設定有誤，程式會顯示錯誤訊息且不會覆蓋此檔案。

translation:
  provider: "ct2-nllb"

  local_llm:
    backend: "ollama"
    base_url: "http://localhost:11434/v1"
    model: "qwen3:4b"
    api_key: ""
    api_key_env: ""
    credential_key: ""
    timeout: 120
    max_retries: 2
    temperature: 0.2
    disable_thinking: true

  ct2_nllb:
    model_path: "./models/nllb"
    model_repo: "JustFrederik/nllb-200-distilled-600M-ct2-int8"
    device: "auto"
    compute_type: "int8"
    fallback_to_cpu: true
    beam_size: 4
    max_decoding_length: 256
    cpu_threads: 0

  source_language: "ko"
  target_language: "zh-TW"
  batch_size: 10
  context_before: 2
  context_after: 2
"""


class CredentialStore(Protocol):
    """Reserved secret lookup protocol (e.g. for future Windows Credential Manager)."""

    def get_secret(self, key: str) -> str | None:
        ...


class NullCredentialStore:
    """Default no-op credential store placeholder until cloud credential storage is needed."""

    def get_secret(self, key: str) -> str | None:
        return None


def resolve_api_key(
    config: LocalLLMConfig,
    *,
    credential_store: CredentialStore | None = None,
) -> str:
    """Resolve an API key safely without encouraging plaintext cloud keys in YAML.

    Lookup order:
    1. Environment variable named by `api_key_env` (if configured).
    2. Secure credential store lookup by `credential_key` (reserved for Windows Credential Manager).
    3. Direct `api_key` value (expected to stay empty for local services).
    """
    env_name = config.api_key_env.strip()
    if env_name:
        env_val = os.environ.get(env_name, "").strip()
        if env_val:
            return env_val

    cred_key = config.credential_key.strip()
    if cred_key and credential_store is not None:
        secret = credential_store.get_secret(cred_key)
        if secret:
            return secret.strip()

    return config.api_key.strip()


def _resolve_config_path(raw_path: str | Path, *, base_dir: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def _expect_mapping(value: Any, field_name: str, config_path: Path) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(
            f"設定檔「{config_path}」的欄位 `{field_name}` 必須是物件（mapping），目前為 {type(value).__name__}。"
        )
    return value


def _expect_str(
    mapping: dict[str, Any],
    key: str,
    default: str,
    *,
    config_path: Path,
    allow_empty: bool = False,
) -> str:
    if key not in mapping or mapping[key] is None:
        return default
    val = mapping[key]
    if not isinstance(val, str):
        raise ConfigError(
            f"設定檔「{config_path}」的欄位 `{key}` 必須是字串，目前為 {type(val).__name__}。"
        )
    cleaned = val.strip()
    if not cleaned and not allow_empty:
        raise ConfigError(f"設定檔「{config_path}」的欄位 `{key}` 不可為空白字串。")
    return cleaned


def _expect_int(
    mapping: dict[str, Any],
    key: str,
    default: int,
    *,
    config_path: Path,
    minimum: int = 0,
) -> int:
    if key not in mapping or mapping[key] is None:
        return default
    val = mapping[key]
    if isinstance(val, bool) or not isinstance(val, int):
        raise ConfigError(
            f"設定檔「{config_path}」的欄位 `{key}` 必須是整數，目前為 {val!r}。"
        )
    if val < minimum:
        raise ConfigError(
            f"設定檔「{config_path}」的欄位 `{key}` 不可小於 {minimum}，目前為 {val}。"
        )
    return val


def _expect_float(
    mapping: dict[str, Any],
    key: str,
    default: float,
    *,
    config_path: Path,
    minimum: float = 0.0,
    exclusive_min: bool = False,
) -> float:
    if key not in mapping or mapping[key] is None:
        return default
    val = mapping[key]
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise ConfigError(
            f"設定檔「{config_path}」的欄位 `{key}` 必須是數值，目前為 {val!r}。"
        )
    num = float(val)
    if (exclusive_min and num <= minimum) or (not exclusive_min and num < minimum):
        op = "大於" if exclusive_min else "大於或等於"
        raise ConfigError(
            f"設定檔「{config_path}」的欄位 `{key}` 必須{op} {minimum}，目前為 {val}。"
        )
    return num


def _expect_bool(
    mapping: dict[str, Any],
    key: str,
    default: bool,
    *,
    config_path: Path,
) -> bool:
    if key not in mapping or mapping[key] is None:
        return default
    val = mapping[key]
    if not isinstance(val, bool):
        raise ConfigError(
            f"設定檔「{config_path}」的欄位 `{key}` 必須是布林值（true/false），目前為 {val!r}。"
        )
    return val


class ConfigManager:
    """Loads, validates, and auto-initializes `config/config.yaml` safely."""

    def __init__(
        self,
        config_path: str | Path | None = None,
        *,
        project_root: Path | None = None,
    ) -> None:
        self._project_root = (project_root or PROJECT_ROOT).expanduser().resolve()
        if config_path is not None:
            self._config_path = Path(config_path).expanduser().resolve()
        else:
            self._config_path = (
                DEFAULT_CONFIG_PATH
                if project_root is None
                else self._project_root / "config" / "config.yaml"
            ).resolve()

    @property
    def config_path(self) -> Path:
        """Return the resolved path to the managed YAML configuration file."""
        return self._config_path

    def ensure_config_exists(self) -> bool:
        """Create default `config.yaml` only when missing. Never overwrites existing files.

        Returns:
            True if a new default config file was created, False if it already existed.
        """
        if self._config_path.exists():
            if self._config_path.is_dir():
                raise ConfigError(f"設定檔路徑不可為資料夾：{self._config_path}")
            return False

        try:
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            self._config_path.write_text(DEFAULT_CONFIG_YAML, encoding="utf-8")
        except OSError as exc:
            raise ConfigError(
                f"無法建立預設設定檔「{self._config_path}」：{exc}"
            ) from exc
        return True

    def load(self, *, auto_create: bool = True) -> TranslationConfig:
        """Load and validate translation configuration from disk."""
        if auto_create:
            self.ensure_config_exists()
        elif not self._config_path.exists():
            raise ConfigError(f"找不到設定檔：{self._config_path}")

        if self._config_path.is_dir():
            raise ConfigError(f"設定檔路徑不可為資料夾：{self._config_path}")

        try:
            raw_text = self._config_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigError(
                f"無法讀取設定檔「{self._config_path}」：{exc}"
            ) from exc

        try:
            parsed = yaml.safe_load(raw_text)
        except yaml.YAMLError as exc:
            raise ConfigError(
                f"設定檔「{self._config_path}」YAML 語法錯誤（系統不會覆蓋您的檔案，請修正後重試）：{exc}"
            ) from exc

        if parsed is None:
            raise ConfigError(
                f"設定檔「{self._config_path}」為空白，請參考 config/config.example.yaml 填入 `translation` 區段。"
            )
        if not isinstance(parsed, dict):
            raise ConfigError(
                f"設定檔「{self._config_path}」頂層結構必須是 YAML 物件（mapping）。"
            )
        if "translation" not in parsed:
            raise ConfigError(
                f"設定檔「{self._config_path}」缺少必要區段 `translation`，請參考 config/config.example.yaml。"
            )

        return self._parse_translation_section(
            _expect_mapping(parsed["translation"], "translation", self._config_path)
        )

    def _parse_translation_section(
        self, section: dict[str, Any]
    ) -> TranslationConfig:
        cfg_path = self._config_path
        provider = _expect_str(
            section, "provider", DEFAULT_PROVIDER, config_path=cfg_path
        ).lower()
        if provider not in SUPPORTED_PROVIDERS:
            supported = ", ".join(SUPPORTED_PROVIDERS)
            raise ConfigError(
                f"設定檔「{cfg_path}」的 `translation.provider` 值「{provider}」無效，僅支援：{supported}。"
            )

        local_llm_raw = _expect_mapping(
            section.get("local_llm"), "translation.local_llm", cfg_path
        )
        base_url = _expect_str(
            local_llm_raw,
            "base_url",
            DEFAULT_LOCAL_LLM_BASE_URL,
            config_path=cfg_path,
        )
        if not (base_url.startswith("http://") or base_url.startswith("https://")):
            raise ConfigError(
                f"設定檔「{cfg_path}」的 `translation.local_llm.base_url` 必須以 http:// 或 https:// 開頭，目前為「{base_url}」。"
            )

        local_llm = LocalLLMConfig(
            backend=_expect_str(
                local_llm_raw,
                "backend",
                DEFAULT_LOCAL_LLM_BACKEND,
                config_path=cfg_path,
            ).lower(),
            base_url=base_url.rstrip("/"),
            model=_expect_str(
                local_llm_raw,
                "model",
                DEFAULT_LOCAL_LLM_MODEL,
                config_path=cfg_path,
            ),
            api_key=_expect_str(
                local_llm_raw,
                "api_key",
                "",
                config_path=cfg_path,
                allow_empty=True,
            ),
            api_key_env=_expect_str(
                local_llm_raw,
                "api_key_env",
                "",
                config_path=cfg_path,
                allow_empty=True,
            ),
            credential_key=_expect_str(
                local_llm_raw,
                "credential_key",
                "",
                config_path=cfg_path,
                allow_empty=True,
            ),
            timeout=_expect_float(
                local_llm_raw,
                "timeout",
                float(DEFAULT_LOCAL_LLM_TIMEOUT),
                config_path=cfg_path,
                minimum=0.0,
                exclusive_min=True,
            ),
            max_retries=_expect_int(
                local_llm_raw,
                "max_retries",
                2,
                config_path=cfg_path,
                minimum=0,
            ),
            temperature=_expect_float(
                local_llm_raw,
                "temperature",
                0.2,
                config_path=cfg_path,
                minimum=0.0,
            ),
            disable_thinking=_expect_bool(
                local_llm_raw,
                "disable_thinking",
                True,
                config_path=cfg_path,
            ),
        )

        nllb_raw = _expect_mapping(
            section.get("ct2_nllb"), "translation.ct2_nllb", cfg_path
        )
        raw_model_path = _expect_str(
            nllb_raw, "model_path", "./models/nllb", config_path=cfg_path
        )
        nllb_device = _expect_str(
            nllb_raw, "device", "auto", config_path=cfg_path
        ).lower()
        if nllb_device not in SUPPORTED_NLLB_DEVICES:
            raise ConfigError(
                f"設定檔「{cfg_path}」的 `translation.ct2_nllb.device` 值「{nllb_device}」無效，"
                f"僅支援：{', '.join(SUPPORTED_NLLB_DEVICES)}。"
            )

        ct2_nllb = CT2NLLBConfig(
            model_path=_resolve_config_path(
                raw_model_path, base_dir=self._project_root
            ),
            model_repo=_expect_str(
                nllb_raw,
                "model_repo",
                DEFAULT_NLLB_MODEL_REPO,
                config_path=cfg_path,
            ),
            device=nllb_device,
            compute_type=_expect_str(
                nllb_raw, "compute_type", "int8", config_path=cfg_path
            ),
            fallback_to_cpu=_expect_bool(
                nllb_raw, "fallback_to_cpu", True, config_path=cfg_path
            ),
            beam_size=_expect_int(
                nllb_raw, "beam_size", 4, config_path=cfg_path, minimum=1
            ),
            max_decoding_length=_expect_int(
                nllb_raw,
                "max_decoding_length",
                256,
                config_path=cfg_path,
                minimum=16,
            ),
            cpu_threads=_expect_int(
                nllb_raw, "cpu_threads", 0, config_path=cfg_path, minimum=0
            ),
        )

        return TranslationConfig(
            provider=provider,
            local_llm=local_llm,
            ct2_nllb=ct2_nllb,
            source_language=_expect_str(
                section,
                "source_language",
                DEFAULT_SOURCE_LANGUAGE,
                config_path=cfg_path,
            ),
            target_language=_expect_str(
                section,
                "target_language",
                DEFAULT_TARGET_LANGUAGE,
                config_path=cfg_path,
            ),
            batch_size=_expect_int(
                section,
                "batch_size",
                DEFAULT_BATCH_SIZE,
                config_path=cfg_path,
                minimum=1,
            ),
            context_before=_expect_int(
                section,
                "context_before",
                DEFAULT_CONTEXT_BEFORE,
                config_path=cfg_path,
                minimum=0,
            ),
            context_after=_expect_int(
                section,
                "context_after",
                DEFAULT_CONTEXT_AFTER,
                config_path=cfg_path,
                minimum=0,
            ),
        )
