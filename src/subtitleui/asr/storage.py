"""JSON persistence helpers for raw ASR transcription results."""

from __future__ import annotations

import json
from pathlib import Path

from ..paths import default_transcript_output_path
from .errors import TranscriptionError
from .models import TranscriptionResult


def save_transcript_json(
    result: TranscriptionResult,
    output_path: str | Path | None = None,
) -> Path:
    """Save a TranscriptionResult to a UTF-8 JSON file.

    When `output_path` is omitted, defaults to
    `outputs/transcription/<source_stem>.transcript.json`.
    Parent directories are created automatically if they do not exist.
    """
    resolved_output = (
        output_path
        if output_path is not None
        else default_transcript_output_path(result.metadata.source)
    )
    target = Path(resolved_output).expanduser().resolve()
    if target.exists() and target.is_dir():
        raise TranscriptionError(f"輸出 JSON 路徑不可為資料夾：{target}")

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = result.to_dict()
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise TranscriptionError(f"無法儲存逐字稿 JSON 檔案「{target}」：{exc}") from exc

    return target


def load_transcript_json(input_path: str | Path) -> TranscriptionResult:
    """Load and validate a TranscriptionResult from a JSON file."""
    source = Path(input_path).expanduser().resolve()
    if not source.exists() or not source.is_file():
        raise TranscriptionError(f"找不到逐字稿 JSON 檔案：{source}")

    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TranscriptionError(f"讀取逐字稿 JSON 失敗「{source}」：{exc}") from exc

    if not isinstance(raw, dict) or "metadata" not in raw or "segments" not in raw:
        raise TranscriptionError(f"逐字稿 JSON 格式不正確（缺少 metadata 或 segments）：{source}")

    try:
        return TranscriptionResult.from_dict(raw, output_path=source)
    except (KeyError, TypeError, ValueError) as exc:
        raise TranscriptionError(f"解析逐字稿 JSON 欄位失敗「{source}」：{exc}") from exc
