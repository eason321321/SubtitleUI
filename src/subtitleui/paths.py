"""Centralized filesystem path configuration for SubtitleUI."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
CONFIG_DIR: Path = PROJECT_ROOT / "config"
DEFAULT_CONFIG_PATH: Path = CONFIG_DIR / "config.yaml"
DEFAULT_CONFIG_EXAMPLE_PATH: Path = CONFIG_DIR / "config.example.yaml"
DOWNLOAD_DIR: Path = PROJECT_ROOT / "downloads"
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"
AUDIO_OUTPUT_DIR: Path = OUTPUT_DIR / "audio"
TRANSCRIPTION_DIR: Path = OUTPUT_DIR / "transcription"
TRANSLATION_DIR: Path = OUTPUT_DIR / "translation"
TEMP_DIR: Path = PROJECT_ROOT / "temp"
MODELS_DIR: Path = PROJECT_ROOT / "models"
NLLB_MODEL_DIR: Path = MODELS_DIR / "nllb"


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if missing and return its resolved Path."""
    target = Path(path).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    return target


def ensure_runtime_dirs() -> None:
    """Ensure all standard runtime data directories exist."""
    for directory in (
        DOWNLOAD_DIR,
        OUTPUT_DIR,
        AUDIO_OUTPUT_DIR,
        TRANSCRIPTION_DIR,
        TRANSLATION_DIR,
        TEMP_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def transcript_stem(input_path: str | Path) -> str:
    """Return the clean media stem from an ASR transcript filename.

    Strips the double `.transcript.json` suffix when present so outputs become
    `<stem>.zh-TW.translation.json` rather than `<stem>.transcript.zh-TW.translation.json`.
    """
    name = Path(input_path).name
    lower_name = name.lower()
    if lower_name.endswith(".transcript.json"):
        return name[: -len(".transcript.json")]
    if lower_name.endswith(".json"):
        return name[: -len(".json")]
    return Path(input_path).stem


def default_translation_output_path(
    transcript_path: str | Path,
    *,
    target_language: str = "zh-TW",
    translation_dir: Path | None = None,
) -> Path:
    """Return the default translation JSON path under outputs/translation/."""
    target_dir = Path(translation_dir or TRANSLATION_DIR).expanduser().resolve()
    stem = transcript_stem(transcript_path)
    lang_tag = target_language.strip() or "zh-TW"
    return target_dir / f"{stem}.{lang_tag}.translation.json"


def resolve_transcript_input_path(
    transcript_path: str | Path,
    *,
    transcription_dir: Path | None = None,
) -> Path:
    """Resolve an input transcript path, checking outputs/transcription/ for relative paths."""
    raw = Path(transcript_path).expanduser()
    if raw.exists() or raw.is_absolute():
        return raw.resolve()

    default_transcriptions = Path(
        transcription_dir or TRANSCRIPTION_DIR
    ).expanduser().resolve()
    candidate = default_transcriptions / raw
    if candidate.exists():
        return candidate.resolve()

    if transcription_dir is None:
        project_candidate = PROJECT_ROOT / raw
        if project_candidate.exists():
            return project_candidate.resolve()

    return raw.resolve()


def default_transcript_output_path(
    input_path: str | Path,
    *,
    transcription_dir: Path | None = None,
) -> Path:
    """Return the default ASR JSON path under outputs/transcription/."""
    target_dir = Path(transcription_dir or TRANSCRIPTION_DIR).expanduser().resolve()
    source = Path(input_path)
    return target_dir / f"{source.stem}.transcript.json"


def default_audio_output_path(
    input_path: str | Path,
    *,
    audio_output_dir: Path | None = None,
    suffix: str = ".wav",
) -> Path:
    """Return the default processed audio path under outputs/audio/."""
    target_dir = Path(audio_output_dir or AUDIO_OUTPUT_DIR).expanduser().resolve()
    source = Path(input_path)
    normalized_suffix = suffix if suffix.startswith(".") else f".{suffix}"
    return target_dir / f"{source.stem}{normalized_suffix}"


def resolve_media_input_path(
    media_path: str | Path,
    *,
    download_dir: Path | None = None,
) -> Path:
    """Resolve an input media path, checking downloads/ for relative paths."""
    raw = Path(media_path).expanduser()
    if raw.exists() or raw.is_absolute():
        return raw.resolve()

    default_downloads = Path(download_dir or DOWNLOAD_DIR).expanduser().resolve()
    candidate = default_downloads / raw
    if candidate.exists():
        return candidate.resolve()

    if download_dir is None:
        project_candidate = PROJECT_ROOT / raw
        if project_candidate.exists():
            return project_candidate.resolve()

    return raw.resolve()
