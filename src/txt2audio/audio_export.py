"""Xuất WAV (soundfile) hoặc MP3 (ffmpeg libmp3lame)."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf

OutputFormat = Literal["wav", "mp3"]


def resolve_ffmpeg_executable(explicit_path: str | None) -> str:
    """Trả về đường dẫn thực thi ffmpeg: ưu tiên đường dẫn người dùng, sau đó PATH."""
    if explicit_path and explicit_path.strip():
        p = Path(explicit_path.strip())
        if p.is_file():
            return str(p.resolve())
        raise FileNotFoundError(f"Không tìm thấy file ffmpeg: {p}")

    w = shutil.which("ffmpeg")
    if w:
        return w
    raise RuntimeError(
        "Không tìm thấy ffmpeg trong PATH. "
        "Chọn đường dẫn tới ffmpeg.exe (hoặc binary ffmpeg) trong ứng dụng."
    )


def normalize_output_path(path: Path, fmt: OutputFormat) -> Path:
    if fmt == "mp3":
        return path.with_suffix(".mp3")
    return path.with_suffix(".wav")


def export_audio(
    audio: np.ndarray,
    sample_rate: int,
    dest: Path,
    fmt: OutputFormat,
    *,
    mp3_bitrate_kbps: int = 32,
    ffmpeg_executable: str | None = None,
) -> Path:
    dest = normalize_output_path(dest, fmt)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "wav":
        sf.write(str(dest), audio, sample_rate, subtype="PCM_16")
        return dest.resolve()

    ffmpeg = resolve_ffmpeg_executable(ffmpeg_executable)
    if mp3_bitrate_kbps not in (32, 128):
        raise ValueError("mp3_bitrate_kbps chỉ hỗ trợ 32 hoặc 128.")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = Path(tmp.name)
    try:
        sf.write(str(tmp_wav), audio, sample_rate, subtype="PCM_16")
        cmd = [
            ffmpeg,
            "-y",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(tmp_wav),
            "-codec:a",
            "libmp3lame",
            "-b:a",
            f"{mp3_bitrate_kbps}k",
            str(dest),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip() or f"mã {proc.returncode}"
            raise RuntimeError(f"ffmpeg MP3 thất bại: {err}")
        return dest.resolve()
    finally:
        try:
            tmp_wav.unlink(missing_ok=True)
        except OSError:
            pass
