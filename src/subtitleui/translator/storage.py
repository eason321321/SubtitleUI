"""Read-only Phase 2 ASR JSON loader and safe translation JSON persistence/merge helpers."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

from ..paths import default_translation_output_path, resolve_transcript_input_path
from .errors import TranscriptReadError, TranslationStorageError
from .models import (
    DEFAULT_SOURCE_LANGUAGE,
    ExistingTranslationPolicy,
    SourceSegment,
    SourceTranscript,
    TranslatedSegment,
    TranslationMetadata,
    TranslationResult,
)


def load_asr_transcript(
    input_path: str | Path,
    *,
    transcription_dir: Path | None = None,
) -> SourceTranscript:
    """Load a Phase 2 ASR transcript JSON file (read-only) with stable 0-based indices."""
    source = resolve_transcript_input_path(
        input_path, transcription_dir=transcription_dir
    )
    if not source.exists() or not source.is_file():
        raise TranscriptReadError(f"找不到逐字稿 JSON 檔案：{source}")

    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TranscriptReadError(
            f"讀取逐字稿 JSON 失敗「{source}」：{exc}"
        ) from exc

    if not isinstance(raw, dict):
        raise TranscriptReadError(f"逐字稿 JSON 頂層格式必須是物件：{source}")
    if "metadata" not in raw or "segments" not in raw:
        raise TranscriptReadError(
            f"逐字稿 JSON 格式不正確（缺少 metadata 或 segments）：{source}"
        )
    if not isinstance(raw["metadata"], dict):
        raise TranscriptReadError(f"逐字稿 JSON 的 metadata 欄位格式無效：{source}")
    if not isinstance(raw["segments"], list):
        raise TranscriptReadError(f"逐字稿 JSON 的 segments 欄位必須是陣列：{source}")

    metadata = raw["metadata"]
    source_media = str(metadata.get("source", ""))
    language = str(metadata.get("language") or DEFAULT_SOURCE_LANGUAGE).strip() or DEFAULT_SOURCE_LANGUAGE

    segments: list[SourceSegment] = []
    for idx, item in enumerate(raw["segments"]):
        if not isinstance(item, dict):
            raise TranscriptReadError(
                f"逐字稿 JSON 第 {idx + 1} 筆 segment 格式無效（非物件）：{source}"
            )
        try:
            start = float(item["start"])
            end = float(item["end"])
            text = str(item["text"])
        except (KeyError, TypeError, ValueError) as exc:
            raise TranscriptReadError(
                f"逐字稿 JSON 第 {idx + 1} 筆 segment 缺少必要欄位（start/end/text）或型別錯誤：{exc}"
            ) from exc

        segments.append(
            SourceSegment(
                index=idx,
                start=start,
                end=end,
                text=text,
            )
        )

    return SourceTranscript(
        source_path=source,
        source_media=source_media,
        language=language,
        segments=tuple(segments),
        raw_metadata=dict(metadata),
    )


def load_translation_json(input_path: str | Path) -> TranslationResult:
    """Load an existing Phase 3 translation JSON file."""
    source = Path(input_path).expanduser().resolve()
    if not source.exists() or not source.is_file():
        raise TranslationStorageError(f"找不到翻譯 JSON 檔案：{source}")

    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TranslationStorageError(
            f"讀取翻譯 JSON 失敗「{source}」：{exc}"
        ) from exc

    if not isinstance(raw, dict) or "metadata" not in raw or "segments" not in raw:
        raise TranslationStorageError(
            f"翻譯 JSON 格式不正確（缺少 metadata 或 segments）：{source}"
        )
    if not isinstance(raw["metadata"], dict) or not isinstance(raw["segments"], list):
        raise TranslationStorageError(f"翻譯 JSON 欄位結構無效：{source}")

    try:
        return TranslationResult.from_dict(raw, output_path=source)
    except (KeyError, TypeError, ValueError) as exc:
        raise TranslationStorageError(
            f"解析翻譯 JSON 欄位失敗「{source}」：{exc}"
        ) from exc


def merge_translation_segments(
    existing_segments: Sequence[TranslatedSegment],
    new_segments: Sequence[TranslatedSegment],
    *,
    source_segments: Sequence[SourceSegment] | None = None,
) -> tuple[TranslatedSegment, ...]:
    """Merge newly translated segments into existing translations by 0-based index.

    - Previously translated segments whose indices are not in `new_segments` are preserved.
    - Indices present in `new_segments` replace the older translations for those indices.
    - Resulting segments are strictly ordered by ascending 0-based index.
    """
    merged: dict[int, TranslatedSegment] = {
        seg.index: seg for seg in existing_segments
    }
    for seg in new_segments:
        merged[seg.index] = seg

    if source_segments is not None:
        source_map = {seg.index: seg for seg in source_segments}
        reconciled: list[TranslatedSegment] = []
        for idx in sorted(merged):
            seg = merged[idx]
            src = source_map.get(idx)
            if src is not None:
                # Always anchor timestamps and source_text to the authoritative ASR segment.
                reconciled.append(
                    TranslatedSegment(
                        index=src.index,
                        start=src.start,
                        end=src.end,
                        source_text=src.text,
                        translated_text=seg.translated_text,
                    )
                )
            else:
                reconciled.append(seg)
        return tuple(reconciled)

    return tuple(merged[idx] for idx in sorted(merged))


def _guard_against_overwriting_asr_json(
    target: Path,
    source_transcript_path: Path | None,
) -> None:
    """Strictly prevent overwriting any Phase 2 raw ASR transcript JSON file."""
    if source_transcript_path is not None:
        src_resolved = source_transcript_path.expanduser().resolve()
        if target == src_resolved:
            raise TranslationStorageError(
                f"安全保護：翻譯輸出路徑不可與原始 Phase 2 逐字稿相同（{target}），請使用獨立的翻譯檔路徑。"
            )
    if target.name.lower().endswith(".transcript.json"):
        raise TranslationStorageError(
            f"安全保護：禁止將翻譯結果寫入 `.transcript.json` 檔案（{target}），以免覆寫 Phase 2 原始逐字稿。"
        )


def _atomic_write_json(target: Path, payload: dict[str, Any]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.stem}.",
        suffix=".tmp",
        dir=str(target.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(serialized)
        os.replace(tmp_path, target)
    except OSError as exc:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise TranslationStorageError(
            f"無法寫入翻譯 JSON 檔案「{target}」：{exc}"
        ) from exc


def save_translation_json(
    result: TranslationResult,
    output_path: str | Path | None = None,
    *,
    source_transcript_path: Path | None = None,
    existing_policy: ExistingTranslationPolicy = ExistingTranslationPolicy.MERGE,
    source_segments: Sequence[SourceSegment] | None = None,
) -> TranslationResult:
    """Save a TranslationResult to UTF-8 JSON with merge support and ASR protection.

    Rules when `output_path` already exists:
    - `MERGE` / `SKIP_EXISTING`: Load existing translations and merge by 0-based `index`,
      preserving previously translated segments not included in this run.
    - `OVERWRITE`: Replace the translation file completely with only `result.segments`.
    """
    resolved_output = (
        output_path
        if output_path is not None
        else default_translation_output_path(
            source_transcript_path or result.metadata.source_transcript,
            target_language=result.metadata.target_language,
        )
    )
    target = Path(resolved_output).expanduser().resolve()
    if target.exists() and target.is_dir():
        raise TranslationStorageError(f"翻譯輸出路徑不可為資料夾：{target}")

    _guard_against_overwriting_asr_json(target, source_transcript_path)

    final_segments = result.segments
    if (
        existing_policy in (ExistingTranslationPolicy.MERGE, ExistingTranslationPolicy.SKIP_EXISTING)
        and target.exists()
    ):
        existing = load_translation_json(target)
        if (
            existing.metadata.source_transcript
            and result.metadata.source_transcript
            and existing.metadata.source_transcript != result.metadata.source_transcript
        ):
            raise TranslationStorageError(
                f"既有翻譯檔「{target}」對應的逐字稿來源為「{existing.metadata.source_transcript}」，"
                f"與目前來源「{result.metadata.source_transcript}」不符，已中止合併以避免混雜資料。"
                "若確定覆寫，請改用 --on-exists overwrite 或指定新的輸出路徑。"
            )
        final_segments = merge_translation_segments(
            existing.segments,
            result.segments,
            source_segments=source_segments,
        )

    updated_metadata = TranslationMetadata(
        source_transcript=result.metadata.source_transcript,
        backend=result.metadata.backend,
        translation_model=result.metadata.translation_model,
        source_language=result.metadata.source_language,
        target_language=result.metadata.target_language,
        selection_mode=result.metadata.selection_mode,
        translated_at=result.metadata.translated_at,
        elapsed_seconds=result.metadata.elapsed_seconds,
        device=result.metadata.device,
        total_source_segments=result.metadata.total_source_segments,
        translated_segments_count=len(final_segments),
    )
    final_result = TranslationResult(
        metadata=updated_metadata,
        segments=final_segments,
        output_path=target,
    )

    _atomic_write_json(target, final_result.to_dict())
    return final_result
