"""Timestamp parsing, formatting, range validation, and offset helpers."""

from __future__ import annotations

import math
import re

from .errors import InvalidTimeRangeError
from .models import TimeRange

_HMS_PATTERN = re.compile(r"^(\d+):(\d{1,2}):(\d{1,2}(?:\.\d+)?)$")
_MS_PATTERN = re.compile(r"^(\d+):(\d{1,2}(?:\.\d+)?)$")
_NUMERIC_PATTERN = re.compile(r"^\d+(?:\.\d+)?$")


def parse_timestamp(value: str | int | float) -> float:
    """Parse a timestamp into non-negative seconds.

    Supported formats:
    - Numeric seconds (int, float, or numeric string like "750" or "750.5")
    - MM:SS or MM:SS.mmm
    - HH:MM:SS or HH:MM:SS.mmm

    Raises:
        InvalidTimeRangeError: If the value is negative, malformed, or out of range.
    """
    if isinstance(value, bool):
        raise InvalidTimeRangeError(f"無效的時間格式：{value!r}")

    if isinstance(value, (int, float)):
        seconds = float(value)
        if not math.isfinite(seconds):
            raise InvalidTimeRangeError(f"時間數值無效：{value!r}")
        if seconds < 0:
            raise InvalidTimeRangeError(f"時間不可小於 0 秒：{value}")
        return round(seconds, 3)

    if not isinstance(value, str):
        raise InvalidTimeRangeError(f"不支援的時間型別：{type(value).__name__}")

    cleaned = value.strip()
    if not cleaned:
        raise InvalidTimeRangeError("時間不可為空白。")

    if cleaned.startswith("-"):
        raise InvalidTimeRangeError(f"時間不可小於 0：{cleaned}")

    hms_match = _HMS_PATTERN.match(cleaned)
    if hms_match:
        hours = int(hms_match.group(1))
        minutes = int(hms_match.group(2))
        secs = float(hms_match.group(3))
        if minutes >= 60 or secs >= 60.0:
            raise InvalidTimeRangeError(
                f"時間格式錯誤（分與秒需小於 60）：{cleaned}"
            )
        return round(hours * 3600 + minutes * 60 + secs, 3)

    ms_match = _MS_PATTERN.match(cleaned)
    if ms_match:
        minutes = int(ms_match.group(1))
        secs = float(ms_match.group(2))
        if secs >= 60.0:
            raise InvalidTimeRangeError(
                f"時間格式錯誤（秒數需小於 60）：{cleaned}"
            )
        return round(minutes * 60 + secs, 3)

    if _NUMERIC_PATTERN.match(cleaned):
        seconds = float(cleaned)
        if not math.isfinite(seconds):
            raise InvalidTimeRangeError(f"時間數值無效：{cleaned}")
        return round(seconds, 3)

    raise InvalidTimeRangeError(
        f"無法解析時間格式「{cleaned}」，請使用 HH:MM:SS（例如 00:12:30）、MM:SS 或秒數。"
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


def resolve_time_range(
    start_time: float | str | None,
    end_time: float | str | None,
    media_duration: float,
) -> tuple[TimeRange, bool]:
    """Validate and resolve a full-media or custom time range.

    Args:
        start_time: Optional start timestamp (None defaults to 0.0).
        end_time: Optional end timestamp (None defaults to media_duration).
        media_duration: Total duration of the original media file in seconds.

    Returns:
        A tuple of ``(TimeRange, is_custom_range)`` where ``is_custom_range`` is
        False when both start_time and end_time are omitted (Mode A: full media).

    Raises:
        InvalidTimeRangeError: If the time format or boundaries are invalid.
    """
    if not math.isfinite(media_duration) or media_duration <= 0:
        raise InvalidTimeRangeError(
            f"媒體檔案長度無效（{media_duration} 秒），無法設定辨識區間。"
        )

    duration_rounded = round(float(media_duration), 3)

    if start_time is None and end_time is None:
        return TimeRange(start=0.0, end=duration_rounded), False

    resolved_start = 0.0 if start_time is None else parse_timestamp(start_time)
    resolved_end = (
        duration_rounded if end_time is None else parse_timestamp(end_time)
    )

    if resolved_start < 0:
        raise InvalidTimeRangeError("開始時間不可小於 0。")

    if resolved_start >= duration_rounded:
        raise InvalidTimeRangeError(
            f"開始時間（{format_timestamp(resolved_start)}）超出影片總長度（{format_timestamp(duration_rounded)}）。"
        )

    if resolved_end > duration_rounded + 1e-6:
        raise InvalidTimeRangeError(
            f"結束時間（{format_timestamp(resolved_end)}）超過影片總長度（{format_timestamp(duration_rounded)}）。"
        )

    if resolved_end <= resolved_start:
        if resolved_end == resolved_start:
            raise InvalidTimeRangeError(
                f"開始時間與結束時間不可相同（皆為 {format_timestamp(resolved_start)}）。"
            )
        raise InvalidTimeRangeError(
            f"結束時間（{format_timestamp(resolved_end)}）必須大於開始時間（{format_timestamp(resolved_start)}）。"
        )

    clamped_end = min(resolved_end, duration_rounded)
    return TimeRange(start=resolved_start, end=clamped_end), True


def apply_timestamp_offset(segment_timestamp: float, selected_start_time: float) -> float:
    """Shift a clip-relative segment timestamp back to the original media timeline.

    Formula:
        original_timestamp = segment_timestamp + selected_start_time
    """
    return round(float(segment_timestamp) + float(selected_start_time), 3)
