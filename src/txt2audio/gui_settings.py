"""Đọc/ghi preset cài đặt GUI ra file .ini (nhiều file, do người dùng chọn)."""

from __future__ import annotations

from configparser import ConfigParser
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from txt2audio.synthesis_tuning import SynthesisTuning

PRESETS_SUBDIR = "settings"

# voice_preset_id trong INI
VOICE_INI_DEFAULT = "__default__"
VOICE_INI_REF = "__ref__"
VOICE_SENTINELS = frozenset({VOICE_INI_DEFAULT, VOICE_INI_REF})


def default_presets_dir(base: Path | None = None) -> Path:
    d = (base or Path.cwd()).resolve() / PRESETS_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_preset_ini_files(base: Path | None = None) -> list[Path]:
    d = default_presets_dir(base)
    return sorted(d.glob("*.ini"), key=lambda p: p.stat().st_mtime, reverse=True)


@dataclass
class BenchmarkIniContext:
    """Ngữ cảnh GUI lúc chạy benchmark (ghi kèm mỗi .ini cặp WAV)."""

    txt_path: str = ""
    out_dir: str = ""
    model_dir: str = ""
    model_file: str = ""
    voice_preset_id: str = ""
    ref_audio: str = ""
    ffmpeg_path: str = ""
    out_format: str = "wav"
    mp3_bitrate: int = 32
    benchmark_sample_chars: int = 500


@dataclass
class LoadResult:
    path: Path
    had_tuning: bool
    pending_voice_preset_id: str | None
    profile_name: str


def _section(cp: ConfigParser, name: str) -> bool:
    return cp.has_section(name) and bool(cp.options(name))


def _get(cp: ConfigParser, section: str, key: str, default: str = "") -> str:
    if not cp.has_section(section):
        return default
    return cp.get(section, key, fallback=default)


def _get_bool(cp: ConfigParser, section: str, key: str, default: bool) -> bool:
    if not cp.has_section(section):
        return default
    return cp.getboolean(section, key, fallback=default)


def _get_int(cp: ConfigParser, section: str, key: str, default: int) -> int:
    if not cp.has_section(section):
        return default
    return cp.getint(section, key, fallback=default)


def _get_float(cp: ConfigParser, section: str, key: str, default: float) -> float:
    if not cp.has_section(section):
        return default
    return cp.getfloat(section, key, fallback=default)


def _resolve_voice_preset_id(app: Any) -> str:
    if app.ref_audio.get().strip():
        return VOICE_INI_REF
    tok = app._selected_preset_id()
    if tok:
        return str(tok)
    idx = app.voice_combo.current()
    if 0 <= idx < len(app.voice_combo_values):
        label, tok2 = app.voice_combo_values[idx]
        if tok2 is None and "Mặc định" in label:
            return VOICE_INI_DEFAULT
    return ""


def tuning_section_dict(
    tuning: SynthesisTuning,
    *,
    benchmark_sample_chars: int | None = None,
) -> dict[str, str]:
    d = {
        "segment_chars": str(int(tuning.segment_chars)),
        "infer_max_chars": str(int(tuning.infer_max_chars)),
        "silence_between_s": str(float(tuning.silence_between_s)),
        "crossfade_between_s": str(float(tuning.crossfade_between_s)),
        "infer_silence_p": str(float(tuning.infer_silence_p)),
        "infer_crossfade_p": str(float(tuning.infer_crossfade_p)),
        "temperature": str(float(tuning.temperature)),
        "top_k": str(int(tuning.top_k)),
        "skip_normalize": "true" if tuning.skip_normalize else "false",
        "show_infer_progress": "true" if tuning.show_infer_progress else "false",
        "turbo_max_tokens": str(int(tuning.turbo_max_tokens)),
    }
    if benchmark_sample_chars is not None:
        d["benchmark_sample_chars"] = str(int(benchmark_sample_chars))
    return d


def export_benchmark_variant_ini(
    ini_path: Path,
    tuning: SynthesisTuning,
    *,
    variant_id: str,
    description: str,
    paired_wav: str,
    is_turbo: bool,
    ctx: BenchmarkIniContext | None = None,
) -> Path:
    """Ghi .ini cùng tên với WAV benchmark — tải lại bằng «Tải preset»."""
    path = ini_path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    cp = ConfigParser()
    stem = path.stem
    cp["meta"] = {
        "name": stem,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "app": "Txt2Audio",
        "source": "benchmark",
    }
    cp["benchmark"] = {
        "variant_id": variant_id,
        "description": description,
        "paired_wav": paired_wav,
        "backend": "turbo" if is_turbo else "standard",
        "voice_summary": tuning.voice_params_summary(is_turbo=is_turbo),
    }
    if ctx is not None:
        cp["paths"] = {
            "txt_path": ctx.txt_path,
            "out_dir": ctx.out_dir,
            "out_name": "",
            "model_dir": ctx.model_dir,
            "ref_audio": ctx.ref_audio,
            "ffmpeg_path": ctx.ffmpeg_path,
        }
        cp["output"] = {
            "format": ctx.out_format,
            "mp3_bitrate": str(int(ctx.mp3_bitrate)),
        }
        cp["model"] = {
            "model_file": ctx.model_file,
            "voice_preset_id": ctx.voice_preset_id,
        }
        cp["tuning"] = tuning_section_dict(
            tuning, benchmark_sample_chars=ctx.benchmark_sample_chars
        )
    else:
        cp["tuning"] = tuning_section_dict(tuning)

    with path.open("w", encoding="utf-8") as f:
        cp.write(f)
    return path


def export_gui_to_configparser(app: Any, *, profile_name: str = "") -> ConfigParser:
    """Ghi toàn bộ trường GUI hiện tại vào ConfigParser."""
    cp = ConfigParser()
    cp["meta"] = {
        "name": profile_name or Path(getattr(app, "_loaded_preset_path", "") or "").stem,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "app": "Txt2Audio",
    }
    cp["paths"] = {
        "txt_path": app.txt_path.get().strip(),
        "out_dir": app.out_dir.get().strip(),
        "out_name": app.out_name.get().strip(),
        "model_dir": app.model_dir.get().strip(),
        "ref_audio": app.ref_audio.get().strip(),
        "ffmpeg_path": app.ffmpeg_path.get().strip(),
    }
    cp["output"] = {
        "format": app.out_format.get().strip() or "wav",
        "mp3_bitrate": str(int(app.mp3_bitrate.get())),
    }
    cp["model"] = {
        "model_file": app.model_file.get().strip(),
        "voice_preset_id": _resolve_voice_preset_id(app),
    }
    t = _tuning_from_app(app)
    cp["tuning"] = tuning_section_dict(
        t, benchmark_sample_chars=int(app.benchmark_sample_chars.get())
    )
    return cp


def _tuning_from_app(app: Any) -> SynthesisTuning:
    return SynthesisTuning(
        segment_chars=int(app.segment_chars.get()),
        infer_max_chars=int(app.infer_max_chars.get()),
        silence_between_s=float(app.silence_s.get()),
        crossfade_between_s=float(app.crossfade_between_s.get()),
        infer_silence_p=float(app.infer_silence_p.get()),
        infer_crossfade_p=float(app.infer_crossfade_p.get()),
        temperature=float(app.temperature.get()),
        top_k=int(app.top_k.get()),
        skip_normalize=bool(app.skip_normalize.get()),
        show_infer_progress=bool(app.show_infer_progress.get()),
        turbo_max_tokens=int(app.turbo_max_tokens.get()),
    )


def benchmark_context_from_app(app: Any) -> BenchmarkIniContext:
    return BenchmarkIniContext(
        txt_path=app.txt_path.get().strip(),
        out_dir=app.out_dir.get().strip(),
        model_dir=app.model_dir.get().strip(),
        model_file=app.model_file.get().strip(),
        voice_preset_id=_resolve_voice_preset_id(app),
        ref_audio=app.ref_audio.get().strip(),
        ffmpeg_path=app.ffmpeg_path.get().strip(),
        out_format=app.out_format.get().strip() or "wav",
        mp3_bitrate=int(app.mp3_bitrate.get()),
        benchmark_sample_chars=int(app.benchmark_sample_chars.get()),
    )


def save_gui_preset(app: Any, ini_path: Path, *, profile_name: str = "") -> Path:
    path = ini_path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    cp = export_gui_to_configparser(app, profile_name=profile_name or path.stem)
    with path.open("w", encoding="utf-8") as f:
        cp.write(f)
    return path


def load_preset_file(path: Path) -> ConfigParser:
    cp = ConfigParser()
    cp.read(path.resolve(), encoding="utf-8")
    return cp


def apply_preset_to_gui(app: Any, cp: ConfigParser, *, source: Path) -> LoadResult:
    """Áp preset lên biến Tkinter; không tự nạp model."""
    had_tuning = _section(cp, "tuning")
    pending_voice: str | None = None

    if _section(cp, "paths"):
        for key, attr in (
            ("txt_path", "txt_path"),
            ("out_dir", "out_dir"),
            ("out_name", "out_name"),
            ("model_dir", "model_dir"),
            ("ref_audio", "ref_audio"),
            ("ffmpeg_path", "ffmpeg_path"),
        ):
            val = _get(cp, "paths", key)
            if val or key in ("ref_audio", "ffmpeg_path", "txt_path"):
                getattr(app, attr).set(val)

    if _section(cp, "output"):
        fmt = _get(cp, "output", "format", "wav")
        if fmt in ("wav", "mp3"):
            app.out_format.set(fmt)
        br = _get_int(cp, "output", "mp3_bitrate", 32)
        if br in (32, 128):
            app.mp3_bitrate.set(br)

    if _section(cp, "model"):
        mf = _get(cp, "model", "model_file")
        if mf:
            app.model_file.set(mf)
        vid = _get(cp, "model", "voice_preset_id")
        if vid in VOICE_SENTINELS or vid:
            pending_voice = vid or None

    if had_tuning:
        app.segment_chars.set(_get_int(cp, "tuning", "segment_chars", 8000))
        app.infer_max_chars.set(_get_int(cp, "tuning", "infer_max_chars", 384))
        app.silence_s.set(_get_float(cp, "tuning", "silence_between_s", 0.2))
        app.crossfade_between_s.set(_get_float(cp, "tuning", "crossfade_between_s", 0.0))
        app.infer_silence_p.set(_get_float(cp, "tuning", "infer_silence_p", 0.15))
        app.infer_crossfade_p.set(_get_float(cp, "tuning", "infer_crossfade_p", 0.0))
        app.temperature.set(_get_float(cp, "tuning", "temperature", 1.0))
        app.top_k.set(_get_int(cp, "tuning", "top_k", 50))
        app.skip_normalize.set(_get_bool(cp, "tuning", "skip_normalize", False))
        app.show_infer_progress.set(_get_bool(cp, "tuning", "show_infer_progress", False))
        app.turbo_max_tokens.set(_get_int(cp, "tuning", "turbo_max_tokens", 1024))
        app.benchmark_sample_chars.set(_get_int(cp, "tuning", "benchmark_sample_chars", 500))
        app._tuning_backend = "preset"

    name = _get(cp, "meta", "name", source.stem)
    app._loaded_preset_path = str(source.resolve())
    app._pending_voice_preset_id = pending_voice

    return LoadResult(
        path=source.resolve(),
        had_tuning=had_tuning,
        pending_voice_preset_id=pending_voice,
        profile_name=name,
    )
