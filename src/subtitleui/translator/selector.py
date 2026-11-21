"""Segment index and time-range selection helpers for Phase 3 translation."""

from __future__ import annotations

import math
import re
from typing import Sequence

from .errors import SelectionError
from .models import SelectionMode, SourceSegment

_HMS_PATTERN = re.compile(r"^(\d+):(\d{1,2}):(\d{1,2}(?:\.\d+)?)$")
_MS_PATTERN = re.compile(r"^(\d+):(\d{1,2}(?:\.\d+)?)$")
_NUMERIC_PATTERN = re.compile(r"^\d+(?:\.\d+)?$")


def parse_timestamp(value: str | int | float) -> float:
    """Parse a timestamp into non-negative seconds (matching Phase 2 formats)."""
    if isinstance(value, bool):
        raise SelectionError(f"無效的時間格式：{value!r}")

    if isinstance(value, (int, float)):
        seconds = float(value)
        if not math.isfinite(seconds):
            raise SelectionError(f"時間數值無效：{value!r}")
        if seconds < 0:
            raise SelectionError(f"時間不可小於 0 秒：{value}")
        return round(seconds, 3)

    if not isinstance(value, str):
        raise SelectionError(f"不支援的時間型別：{type(value).__name__}")

    cleaned = value.strip()
    if not cleaned:
        raise SelectionError("時間不可為空白。")
    if cleaned.startswith("-"):
        raise SelectionError(f"時間不可小於 0：{cleaned}")

    hms_match = _HMS_PATTERN.match(cleaned)
    if hms_match:
        hours = int(hms_match.group(1))
        minutes = int(hms_match.group(2))
        secs = float(hms_match.group(3))
        if minutes >= 60 or secs >= 60.0:
            raise SelectionError(f"時間格式錯誤（分與秒需小於 60）：{cleaned}")
        return round(hours * 3600 + minutes * 60 + secs, 3)

    ms_match = _MS_PATTERN.match(cleaned)
    if ms_match:
        minutes = int(ms_match.group(1))
        secs = float(ms_match.group(2))
        if secs >= 60.0:
            raise SelectionError(f"時間格式錯誤（秒數需小於 60）：{cleaned}")
        return round(minutes * 60 + secs, 3)

    if _NUMERIC_PATTERN.match(cleaned):
        seconds = float(cleaned)
        if not math.isfinite(seconds):
            raise SelectionError(f"時間數值無效：{cleaned}")
        return round(seconds, 3)

    raise SelectionError(
        f"無法解析時間格式「{cleaned}」，請使用 HH:MM:SS（例如 00:00:10）、MM:SS 或秒數。"
    )


def format_timestamp(seconds: float, *, include_millis: bool = False) -> str:
    """Format non-negative seconds as HH:MM:SS or HH:MM:SS.mmm."""
    safe = max(0.0, float(seconds))
    total_ms = round(safe * 1000)
    hours, rem_ms = divmod(total_ms, 3_600_000)
    minutes, rem_ms = divmod(rem_ms, 60_000)
    secs, millis = divmod(rem_ms, 1000)
    if include_millis or millis > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def parse_segment_spec(spec: str, total_segments: int) -> tuple[int, ...]:
    """Parse a 1-based segment selector like `1,3,5-10` into sorted 0-based indices.

    Args:
        spec: User-facing 1-based comma-separated indices and inclusive ranges.
        total_segments: Total number of segments in the source transcript.

    Returns:
        Sorted, deduplicated tuple of 0-based indices.
    """
    if not isinstance(spec, str) or not spec.strip():
        raise SelectionError("指定句子參數（--segments）不可為空白。")
    if total_segments <= 0:
        raise SelectionError("逐字稿沒有任何片段可供選取。")

    selected: set[int] = set()
    parts = [part.strip() for part in spec.split(",")]

    for part in parts:
        if not part:
            raise SelectionError(
                f"指定句子格式錯誤「{spec}」（包含空白區段，範例：1,3,5-10）。"
            )
        if "-" in part:
            bounds = part.split("-")
            if len(bounds) != 2 or not bounds[0].strip().isdigit() or not bounds[1].strip().isdigit():
                raise SelectionError(
                    f"指定句子區間格式錯誤「{part}」，請使用正整數區間（例如 5-10）。"
                )
            start_1 = int(bounds[0].strip())
            end_1 = int(bounds[1].strip())
            if start_1 < 1 or end_1 < 1:
                raise SelectionError(
                    f"句子編號採 1-based，不可小於 1（收到：{part}）。"
                )
            if end_1 < start_1:
                raise SelectionError(
                    f"句子區間結束編號不可小於開始編號（收到：{part}）。"
                )
            if end_1 > total_segments:
                raise SelectionError(
                    f"句子編號 {end_1} 超出逐字稿總句數（共 {total_segments} 句，有效範圍 1-{total_segments}）。"
                )
            for one_based in range(start_1, end_1 + 1):
                selected.add(one_based - 1)
        else:
            if not part.isdigit():
                raise SelectionError(
                    f"無效的句子編號「{part}」，請使用正整數或區間（例如 1,3,5-10）。"
                )
            one_based = int(part)
            if one_based < 1:
                raise SelectionError(
                    f"句子編號採 1-based，不可小於 1（收到：{part}）。"
                )
            if one_based > total_segments:
                raise SelectionError(
                    f"句子編號 {one_based} 超出逐字稿總句數（共 {total_segments} 句，有效範圍 1-{total_segments}）。"
                )
            selected.add(one_based - 1)

    if not selected:
        raise SelectionError(f"未能從「{spec}」解析出任何有效的句子編號。")

    return tuple(sorted(selected))


def normalize_segment_indices(
    indices: Sequence[int],
    total_segments: int,
) -> tuple[int, ...]:
    """Validate a Python API sequence of 0-based segment indices."""
    if not indices:
        raise SelectionError("segment_indices 不可為空。")
    if total_segments <= 0:
        raise SelectionError("逐字稿沒有任何片段可供選取。")

    selected: set[int] = set()
    for idx in indices:
        if isinstance(idx, bool) or not isinstance(idx, int):
            raise SelectionError(f"句子索引必須是整數，收到：{idx!r}")
        if idx < 0 or idx >= total_segments:
            raise SelectionError(
                f"0-based 句子索引 {idx} 超出範圍（有效範圍 0-{total_segments - 1}）。"
            )
        selected.add(idx)

    return tuple(sorted(selected))


def select_segments(
    segments: Sequence[SourceSegment],
    *,
    segments_spec: str | None = None,
    segment_indices: Sequence[int] | None = None,
    start_time: float | str | None = None,
    end_time: float | str | None = None,
) -> tuple[SelectionMode, tuple[SourceSegment, ...]]:
    """Select target segments by full transcript, segment indices, or time range.

    Time range intersection rule:
        ``segment.end > start and segment.start < end``
    """
    has_seg_spec = segments_spec is not None
    has_seg_indices = segment_indices is not None
    has_time_range = start_time is not None or end_time is not None

    active_selectors = sum(
        1 for flag in (has_seg_spec or has_seg_indices, has_time_range) if flag
    )
    if active_selectors > 1:
        raise SelectionError(
            "不可同時指定句子編號（--segments）與時間區間（--start / --end），請擇一使用。"
        )
    if has_seg_spec and has_seg_indices:
        raise SelectionError("不可同時傳入 segments_spec 與 segment_indices。")

    all_segments = tuple(segments)

    if not has_seg_spec and not has_seg_indices and not has_time_range:
        return SelectionMode.FULL, all_segments

    if has_seg_spec:
        assert segments_spec is not None
        chosen_indices = parse_segment_spec(segments_spec, len(all_segments))
        by_index = {seg.index: seg for seg in all_segments}
        return SelectionMode.SEGMENTS, tuple(by_index[i] for i in chosen_indices)

    if has_seg_indices:
        assert segment_indices is not None
        chosen_indices = normalize_segment_indices(segment_indices, len(all_segments))
        by_index = {seg.index: seg for seg in all_segments}
        return SelectionMode.SEGMENTS, tuple(by_index[i] for i in chosen_indices)

    # Time-range mode
    resolved_start = 0.0 if start_time is None else parse_timestamp(start_time)
    resolved_end = (
        math.inf if end_time is None else parse_timestamp(end_time)
    )

    if resolved_end <= resolved_start:
        if resolved_end == resolved_start:
            raise SelectionError(
                f"開始時間與結束時間不可相同（皆為 {format_timestamp(resolved_start)}）。"
            )
        raise SelectionError(
            f"結束時間（{format_timestamp(resolved_end)}）必須大於開始時間（{format_timestamp(resolved_start)}）。"
        )

    matched = tuple(
        seg
        for seg in all_segments
        if seg.end > resolved_start and seg.start < resolved_end
    )
    return SelectionMode.TIME_RANGE, matched
