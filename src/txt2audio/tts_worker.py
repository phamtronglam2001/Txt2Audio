"""Luồng nền: Vieneu TTS theo từng đoạn sách, hủy an toàn giữa các đoạn."""

from __future__ import annotations

import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from vieneu import Vieneu
from vieneu_utils.core_utils import join_audio_chunks

from txt2audio.audio_export import OutputFormat, export_audio
from txt2audio.gguf_loader import (
    TURBO_REPO_ID,
    resolve_backbone_repo_for_gguf,
    resolve_gguf_model_path,
    vieneu_mode_for_repo,
)
from txt2audio.local_assets import ensure_portable_bundle, ensure_turbo_codec
from txt2audio.model_paths import BUNDLE_SUBDIR_BY_REPO, bundle_dir_for_gguf_file
from txt2audio.local_standard_backend import LocalPathStandardVieNeuTTS

try:
    from vieneu.standard import VieNeuTTS as _SdkStandardTTS
    from vieneu.turbo import TurboVieNeuTTS as _SdkTurboTTS
except ImportError:  # pragma: no cover
    _SdkStandardTTS = None  # type: ignore[misc, assignment]
    _SdkTurboTTS = None  # type: ignore[misc, assignment]


def _tts_backend_kind(tts: Any) -> str:
    if isinstance(tts, LocalPathStandardVieNeuTTS):
        return "standard"
    if _SdkStandardTTS is not None and isinstance(tts, _SdkStandardTTS):
        return "standard"
    if _SdkTurboTTS is not None and isinstance(tts, _SdkTurboTTS):
        return "turbo"
    name = type(tts).__name__
    if "Turbo" in name:
        return "turbo"
    if "VieNeu" in name or "Standard" in name:
        return "standard"
    return "unknown"
from txt2audio.synthesis_tuning import SynthesisTuning
from txt2audio.text_chunker import book_segments, read_txt_file

LogFn = Callable[[str], None]
ProgressFn = Callable[[int, int], None]
ModelReadyFn = Callable[[list[tuple[str, str]]], None]
DoneFn = Callable[[str], None]
ErrFn = Callable[[str], None]
FinishedFn = Callable[[], None]
PrepareUiFn = Callable[[Any], None]  # nhận object nhỏ chứa tts/voices/err (worker gọi → GUI schedule main thread)

TORCH_MISSING_MSG = (
    "Repo Q8 đang dùng backend 'standard' và cần thư viện 'torch', "
    "nhưng môi trường hiện chưa có torch.\n"
    "Cách xử lý:\n"
    "1) Cài torch CPU vào env rồi chạy lại, hoặc\n"
    "2) Chuyển sang repo Turbo v2 (không cần standard backend)."
)

CODEC_MISSING_MSG = (
    "Repo Q8 (backend 'standard') cần package 'neucodec' để giải mã audio.\n"
    "Cách xử lý: trong thư mục dự án chạy\n"
    "  uv sync\n"
    "hoặc\n"
    "  uv pip install neucodec\n"
    "Sau đó bấm lại «Nạp model & giọng»."
)


def _infer_segment(
    tts: Any,
    *,
    backbone_repo_id: str,
    text: str,
    voice: Any,
    tuning: SynthesisTuning,
) -> np.ndarray:
    is_turbo = vieneu_mode_for_repo(backbone_repo_id) == "turbo"
    kwargs: dict[str, Any] = {"text": text, "voice": voice, **tuning.infer_kwargs(is_turbo=is_turbo)}
    return tts.infer(**kwargs)


def _init_vieneu(
    backbone_repo_id: str,
    model_path: Path,
    on_log: LogFn,
    *,
    allow_download: bool = True,
) -> Any:
    mode = vieneu_mode_for_repo(backbone_repo_id)
    bundle = bundle_dir_for_gguf_file(model_path)
    ensure_portable_bundle(
        bundle,
        backbone_repo_id,
        allow_download=allow_download,
        on_log=on_log,
    )
    on_log(f"Đang tải model Vieneu mode='{mode}', có thể lâu…")
    try:
        if mode == "standard":
            return LocalPathStandardVieNeuTTS(
                backbone_repo=str(model_path),
                backbone_device="cpu",
                voices_repo_id=backbone_repo_id,
                assets_dir=str(bundle),
            )

        decoder, encoder = ensure_turbo_codec(
            bundle, allow_download=allow_download, on_log=on_log
        )
        return Vieneu(
            mode=mode,
            backbone_repo=str(model_path),
            backbone_filename=model_path.name,
            decoder_repo=str(decoder),
            encoder_repo=str(encoder),
        )
    except ModuleNotFoundError as e:
        if mode == "standard" and "torch" in str(e).lower():
            raise RuntimeError(TORCH_MISSING_MSG) from e
        if mode == "standard" and "neucodec" in str(e).lower():
            raise RuntimeError(CODEC_MISSING_MSG) from e
        raise
    except ImportError as e:
        if mode == "standard" and "neucodec" in str(e).lower():
            raise RuntimeError(CODEC_MISSING_MSG) from e
        if mode == "standard" and "pytorch" in str(e).lower():
            raise RuntimeError(CODEC_MISSING_MSG) from e
        raise


def _repo_id_for_gguf_selection(
    model_cache_dir: str,
    preferred_model_filename: Optional[str],
    on_log: LogFn | None = None,
) -> str:
    """Đoán repo HF từ file .gguf được chọn (không cần combobox loại model)."""
    cache = Path(model_cache_dir)
    if preferred_model_filename:
        return resolve_backbone_repo_for_gguf(cache / preferred_model_filename, on_log)
    if cache.is_dir():
        ggufs = sorted(cache.glob("*.gguf"), key=lambda p: p.name.lower())
        if ggufs:
            return resolve_backbone_repo_for_gguf(ggufs[0], on_log)
    return TURBO_REPO_ID


@dataclass
class PrepareModelResult:
    """Kết quả nạp model (thread worker → GUI)."""

    tts: Any | None = None
    voices: list[tuple[str, str]] | None = None
    error: str | None = None
    effective_repo_id: str | None = None
    prepared_gguf_filename: str | None = None


def run_prepare_model_job(  # noqa: PLR0915 — luồng chuẩn bị tách rõ từng bước
    *,
    model_cache_dir: str,
    preferred_model_filename: Optional[str],
    allow_download_if_missing: bool,
    cancel_event: threading.Event,
    on_log: LogFn,
    on_ui: PrepareUiFn,
) -> None:
    """Chỉ resolve GGUF + khởi tạo Vieneu + liệt kê preset; không đọc TXT."""
    res = PrepareModelResult()
    tts_local: Any | None = None
    try:
        if cancel_event.is_set():
            res.error = "Đã hủy."
            return
        backbone_repo_id = _repo_id_for_gguf_selection(
            model_cache_dir, preferred_model_filename, on_log
        )
        model_path = resolve_gguf_model_path(
            repo_id=backbone_repo_id,
            model_cache_dir=model_cache_dir,
            preferred_model_filename=preferred_model_filename,
            allow_download_if_missing=allow_download_if_missing,
            on_log=on_log,
        )
        if cancel_event.is_set():
            res.error = "Đã hủy."
            return
        effective_repo = resolve_backbone_repo_for_gguf(model_path, on_log)
        tts_local = _init_vieneu(
            effective_repo,
            model_path,
            on_log,
            allow_download=allow_download_if_missing,
        )
        res.effective_repo_id = effective_repo
        sub = BUNDLE_SUBDIR_BY_REPO.get(effective_repo, "vieneu-turbo")
        res.prepared_gguf_filename = f"{sub}/{model_path.name}"
        if cancel_event.is_set():
            try:
                tts_local.close()
            except Exception:
                pass
            res.error = "Đã hủy."
            return
        voices = tts_local.list_preset_voices()
        res.tts = tts_local
        res.voices = voices
        tts_local = None
        for desc, vid in voices:
            on_log(f"  Giọng: {desc}  (id={vid!r})")
    except Exception as e:
        if tts_local is not None:
            try:
                tts_local.close()
            except Exception:
                pass
        res.error = f"{e}\n{traceback.format_exc()}"
    finally:
        on_ui(res)


def run_synthesis_job(
    *,
    txt_path: str,
    out_path: str,
    tuning: SynthesisTuning,
    model_cache_dir: str,
    preferred_model_filename: Optional[str],
    allow_download_if_missing: bool,
    preset_voice_id: Optional[str],
    reference_audio_path: Optional[str],
    cancel_event: threading.Event,
    tts: Optional[Any] = None,
    output_format: OutputFormat = "wav",
    mp3_bitrate_kbps: int = 32,
    ffmpeg_path: Optional[str] = None,
    on_log: LogFn,
    on_progress: ProgressFn,
    on_model_ready: ModelReadyFn,
    on_done: DoneFn,
    on_error: ErrFn,
    on_finished: Optional[FinishedFn] = None,
) -> None:
    try:
        text = read_txt_file(txt_path)
        on_log(f"Đã đọc TXT (UTF-8): {len(text)} ký tự.")
        segments = book_segments(text, tuning.segment_chars)
        if not segments:
            on_error("File TXT rỗng hoặc chỉ có khoảng trắng.")
            return

        on_log(
            f"Tuning: temp={tuning.temperature}, top_k={tuning.top_k}, "
            f"infer_silence={tuning.infer_silence_p}s (Q8), ghép đoạn={tuning.silence_between_s}s"
        )
        on_log(f"Đã chia thành {len(segments)} đoạn (tối đa ~{tuning.segment_chars} ký tự/đoạn).")
        backbone_repo_id = _repo_id_for_gguf_selection(
            model_cache_dir, preferred_model_filename, on_log
        )
        effective_repo = backbone_repo_id
        if tts is None:
            model_path = resolve_gguf_model_path(
                repo_id=backbone_repo_id,
                model_cache_dir=model_cache_dir,
                preferred_model_filename=preferred_model_filename,
                allow_download_if_missing=allow_download_if_missing,
                on_log=on_log,
            )
            effective_repo = resolve_backbone_repo_for_gguf(model_path, on_log)
            tts = _init_vieneu(
                effective_repo,
                model_path,
                on_log,
                allow_download=allow_download_if_missing,
            )
            try:
                voices = tts.list_preset_voices()
                on_model_ready(voices)
                for desc, vid in voices:
                    on_log(f"  Giọng: {desc}  (id={vid!r})")
            except Exception as e:
                on_log(f"(Không liệt kê được preset voices: {e})")
        else:
            on_log("Dùng model Vieneu đã nạp sẵn từ bước chuẩn bị.")
            model_path = resolve_gguf_model_path(
                repo_id=backbone_repo_id,
                model_cache_dir=model_cache_dir,
                preferred_model_filename=preferred_model_filename,
                allow_download_if_missing=False,
                on_log=on_log,
            )
            effective_repo = resolve_backbone_repo_for_gguf(model_path, on_log)
            need_standard = vieneu_mode_for_repo(effective_repo) == "standard"
            loaded = _tts_backend_kind(tts)
            if need_standard and loaded != "standard":
                on_error(
                    "Model đang nạp là Turbo nhưng file GGUF là Q8.\n"
                    "Bấm «Nạp model & giọng» lại (app sẽ tự chọn backend Q8)."
                )
                return
            if not need_standard and loaded != "turbo":
                on_error(
                    "Model đang nạp là Q8 nhưng file GGUF là Turbo.\n"
                    "Bấm «Nạp model & giọng» lại."
                )
                return

        voice: Any = None
        if reference_audio_path:
            p = Path(reference_audio_path)
            if not p.is_file():
                on_error(f"File tham chiếu không tồn tại: {p}")
                return
            on_log(f"Đang mã hóa giọng tham chiếu: {p.name}")
            voice = tts.encode_reference(str(p))
        elif preset_voice_id:
            voice = tts.get_preset_voice(preset_voice_id)

        waves: list[np.ndarray] = []
        total = len(segments)
        sr = int(tts.sample_rate)

        for i, seg in enumerate(segments):
            if cancel_event.is_set():
                on_log("Đã dừng theo yêu cầu.")
                return
            on_log(f"Đoạn {i + 1}/{total} ({len(seg)} ký tự)…")
            w = _infer_segment(
                tts,
                backbone_repo_id=effective_repo,
                text=seg,
                voice=voice,
                tuning=tuning,
            )
            if w is None or (isinstance(w, np.ndarray) and w.size == 0):
                on_log(f"Cảnh báo: đoạn {i + 1} trả về audio rỗng, bỏ qua.")
                on_progress(i + 1, total)
                continue
            waves.append(np.asarray(w, dtype=np.float32))
            on_progress(i + 1, total)

        if cancel_event.is_set():
            on_log("Đã dừng trước khi ghép file.")
            return

        if not waves:
            on_error(
                "Không có sóng âm thanh nào được tạo.\n"
                "Kiểm tra: file GGUF có khớp repo (Q8 ↔ Turbo), giọng đã chọn, nội dung TXT."
            )
            return

        on_log("Đang ghép audio…")
        combined = join_audio_chunks(
            waves,
            sr,
            silence_p=float(tuning.silence_between_s),
            crossfade_p=float(tuning.crossfade_between_s),
        )
        peak = float(np.max(np.abs(combined))) if combined.size else 0.0
        dur_s = combined.size / sr if sr else 0.0
        on_log(f"Audio ghép: {dur_s:.1f}s, peak={peak:.4f}, {len(waves)}/{total} đoạn có tiếng.")
        if peak < 1e-5:
            on_error(
                "Audio gần như im lặng sau khi ghép.\n"
                "Thường do GGUF Q8 nhưng chạy backend Turbo (hoặc ngược lại) — "
                "chọn đúng repo, bấm «Nạp model» lại rồi đọc sách."
            )
            return
        out = Path(out_path)
        br = int(mp3_bitrate_kbps)
        if br not in (32, 128):
            br = 32
        ff = (ffmpeg_path or "").strip() or None
        final = export_audio(
            combined,
            sr,
            out,
            output_format,
            mp3_bitrate_kbps=br,
            ffmpeg_executable=ff,
        )
        size_kb = final.stat().st_size / 1024
        on_log(f"Đã lưu {final.name} ({size_kb:.1f} KB, {dur_s:.1f}s).")
        on_done(str(final))
    except Exception as e:
        on_error(f"{e}\n{traceback.format_exc()}")
    finally:
        if on_finished is not None:
            on_finished()
