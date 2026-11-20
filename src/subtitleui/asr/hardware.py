"""Hardware detection and CUDA-to-CPU fallback helpers for Faster-Whisper."""

from __future__ import annotations

import ctypes
import os
import site
import sys
from pathlib import Path
from typing import Any, Callable

from .errors import ModelLoadError
from .models import HardwareConfig, SUPPORTED_DEVICES

_REGISTERED_DLL_HANDLES: list[Any] = []
_REGISTERED_DLL_DIRS: set[Path] = set()

_WINDOWS_CUDA_DLLS = (
    "cublas64_12.dll",
    "cudnn_ops64_9.dll",
)

_CUDA_ERROR_KEYWORDS = (
    "cuda",
    "cudnn",
    "cublas",
    "libcublas",
    "libcudnn",
    "out of memory",
    "cannot be loaded",
    "could not locate",
    "unsupported device",
    "no cuda",
    "driver",
)


def register_nvidia_dll_directories() -> list[Path]:
    """Register pip-installed NVIDIA runtime DLL folders on Windows.

    When packages such as `nvidia-cublas-cu12` and `nvidia-cudnn-cu12` are
    installed inside the virtual environment, CTranslate2 on Windows requires
    their `bin/` directories to be added to the DLL search path and PATH.
    """
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


def _default_cuda_device_count() -> int:
    import ctranslate2

    return int(ctranslate2.get_cuda_device_count())


def _default_windows_dll_check(dll_name: str) -> bool:
    if sys.platform != "win32":
        return True
    try:
        ctypes.WinDLL(dll_name)
        return True
    except OSError:
        return False


def check_cuda_availability(
    *,
    cuda_device_count_fn: Callable[[], int] | None = None,
    dll_check_fn: Callable[[str], bool] | None = None,
    platform_name: str | None = None,
) -> tuple[bool, str | None]:
    """Check whether CUDA + required runtime DLLs are usable for CTranslate2."""
    active_platform = platform_name or sys.platform
    if active_platform == "win32":
        register_nvidia_dll_directories()

    count_fn = cuda_device_count_fn or _default_cuda_device_count
    try:
        device_count = count_fn()
    except Exception as exc:
        return False, f"無法偵測 CUDA 裝置（{exc}）"

    if device_count <= 0:
        return False, "偵測不到可用的 CUDA GPU"

    if active_platform == "win32":
        checker = dll_check_fn or _default_windows_dll_check
        missing = [dll for dll in _WINDOWS_CUDA_DLLS if not checker(dll)]
        if missing:
            missing_str = ", ".join(missing)
            return (
                False,
                f"缺少 CUDA/cuDNN 執行期函式庫（{missing_str}）",
            )

    return True, None


def resolve_hardware_candidates(
    *,
    preferred_device: str = "auto",
    preferred_compute_type: str | None = None,
    cuda_checker: Callable[[], tuple[bool, str | None]] | None = None,
) -> list[HardwareConfig]:
    """Build an ordered list of hardware configurations to try during model load.

    Strategy for `auto`:
    1. Prefer CUDA + FP16 when a compatible GPU and runtime libraries exist.
    2. Automatically fall back to CPU + INT8 if CUDA is unavailable or fails.
    """
    device = preferred_device.strip().lower()
    if device not in SUPPORTED_DEVICES:
        raise ModelLoadError(
            f"不支援的運算裝置「{preferred_device}」，請選擇 {', '.join(SUPPORTED_DEVICES)}。"
        )

    checker = cuda_checker or check_cuda_availability

    if device == "cpu":
        return [
            HardwareConfig(
                device="cpu",
                compute_type=preferred_compute_type or "int8",
                fallback_used=False,
            )
        ]

    cuda_ok, cuda_reason = checker()

    if device == "cuda":
        if not cuda_ok:
            raise ModelLoadError(
                f"已指定使用 CUDA，但目前無法啟用 GPU：{cuda_reason or '未知原因'}。"
                "請檢查 NVIDIA 驅動、CUDA/cuDNN 函式庫，或改用 --device auto / --device cpu。"
            )
        return [
            HardwareConfig(
                device="cuda",
                compute_type=preferred_compute_type or "float16",
                fallback_used=False,
            )
        ]

    # device == "auto"
    if cuda_ok:
        return [
            HardwareConfig(
                device="cuda",
                compute_type=preferred_compute_type or "float16",
                fallback_used=False,
            ),
            HardwareConfig(
                device="cpu",
                compute_type="int8",
                fallback_used=True,
                fallback_reason="CUDA 初始化失敗，已自動切換為 CPU (INT8)",
            ),
        ]

    reason = f"{cuda_reason}，已自動切換為 CPU (INT8)" if cuda_reason else "CUDA 不可用，已自動切換為 CPU (INT8)"
    return [
        HardwareConfig(
            device="cpu",
            compute_type=preferred_compute_type or "int8",
            fallback_used=True,
            fallback_reason=reason,
        )
    ]


def is_cuda_runtime_error(exc: Exception) -> bool:
    """Return True if an exception appears to stem from CUDA/cuDNN runtime issues."""
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(keyword in text for keyword in _CUDA_ERROR_KEYWORDS)
