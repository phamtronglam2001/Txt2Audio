"""Benchmark giọng: một đoạn TXT → nhiều file audio với các bộ tham số khác nhau."""

from __future__ import annotations

import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from vieneu_utils.core_utils import join_audio_chunks

from txt2audio.audio_export import export_audio
from txt2audio.benchmark_plan import BenchmarkPlan, build_benchmark_variants
from txt2audio.gui_settings import BenchmarkIniContext, export_benchmark_variant_ini
from txt2audio.gguf_loader import vieneu_mode_for_repo
from txt2audio.synthesis_tuning import SynthesisTuning
from txt2audio.text_chunker import book_segments, read_txt_file
from txt2audio.tts_worker import _infer_segment

LogFn = Callable[[str], None]
ProgressFn = Callable[[int, int], None]
DoneFn = Callable[[str], None]
ErrFn = Callable[[str], None]
FinishedFn = Callable[[], None]


def _synthesize_sample(
    text: str,
    *,
    tts: Any,
    voice: Any,
    effective_repo: str,
    tuning: SynthesisTuning,
) -> np.ndarray:
    """Tổng hợp audio cho mẫu văn bản (một hoặc vài đoạn sách)."""
    segments = book_segments(text, tuning.segment_chars)
    if not segments:
        return np.array([], dtype=np.float32)

    waves: list[np.ndarray] = []
    sr = int(tts.sample_rate)
    for seg in segments:
        w = _infer_segment(
            tts,
            backbone_repo_id=effective_repo,
            text=seg,
            voice=voice,
            tuning=tuning,
        )
        if w is None or (isinstance(w, np.ndarray) and w.size == 0):
            continue
        waves.append(np.asarray(w, dtype=np.float32))

    if not waves:
        return np.array([], dtype=np.float32)
    return join_audio_chunks(
        waves,
        sr,
        silence_p=float(tuning.silence_between_s),
        crossfade_p=float(tuning.crossfade_between_s),
    )


def run_benchmark_job(
    *,
    txt_path: str,
    output_dir: Path,
    sample_chars: int,
    tuning_base: SynthesisTuning,
    effective_repo_id: str,
    preset_voice_id: Optional[str],
    reference_audio_path: Optional[str],
    tts: Any,
    cancel_event: Any,
    on_log: LogFn,
    on_progress: ProgressFn,
    on_done: DoneFn,
    on_error: ErrFn,
    on_finished: Optional[FinishedFn] = None,
    ini_context: BenchmarkIniContext | None = None,
    plan: BenchmarkPlan | None = None,
) -> None:
    """Xuất WAV benchmark vào output_dir; không dùng MP3 (tránh gọi ffmpeg nhiều lần)."""
    try:
        raw = read_txt_file(txt_path)
        limit = max(80, int(sample_chars))
        text = raw[:limit].strip()
        if not text:
            on_error("TXT rỗng hoặc quá ngắn.")
            return

        is_turbo = vieneu_mode_for_repo(effective_repo_id) == "turbo"
        bench_plan = plan or BenchmarkPlan.default_for_backend(is_turbo)
        variants = build_benchmark_variants(tuning_base, is_turbo=is_turbo, plan=bench_plan)
        total = len(variants)
        if total == 0:
            on_error("Benchmark: chưa chọn trục tham số nào.")
            return

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        bench_root = output_dir / f"benchmark_{Path(txt_path).stem}_{stamp}"
        bench_root.mkdir(parents=True, exist_ok=True)

        on_log(f"--- Benchmark giọng ({total} biến thể) ---")
        on_log(f"Mẫu: {len(text)} ký tự đầu file (giới hạn {limit}).")
        on_log(f"Thư mục: {bench_root}")

        voice: Any = None
        if reference_audio_path:
            voice = tts.encode_reference(str(reference_audio_path))
        elif preset_voice_id:
            voice = tts.get_preset_voice(preset_voice_id)

        lines: list[str] = [
            f"# Benchmark giọng — {datetime.now().isoformat(timespec='seconds')}",
            f"TXT: {txt_path}",
            f"Mẫu {len(text)} ký tự",
            f"Backend: {'turbo' if is_turbo else 'standard'}",
            "# variant_id\\twav\\tini\\tmô_tả\\ttham_số",
            "",
        ]

        for i, (name, desc, tuning) in enumerate(variants):
            if cancel_event.is_set():
                on_log("Benchmark đã hủy.")
                break
            on_log(f"[{i + 1}/{total}] {name}: {desc}")
            on_log(f"  → {tuning.voice_params_summary(is_turbo=is_turbo)}")

            audio = _synthesize_sample(
                text,
                tts=tts,
                voice=voice,
                effective_repo=effective_repo_id,
                tuning=tuning,
            )
            if audio.size == 0:
                on_log("  Cảnh báo: audio rỗng, bỏ qua.")
                lines.append(f"{name}\tSKIP\t{desc}\t{tuning.voice_params_summary(is_turbo=is_turbo)}")
                on_progress(i + 1, total)
                continue

            stem = tuning.benchmark_wav_stem(name, is_turbo=is_turbo)
            wav_name = f"{stem}.wav"
            ini_name = f"{stem}.ini"
            out_file = bench_root / wav_name
            ini_file = bench_root / ini_name
            export_audio(audio, int(tts.sample_rate), out_file, "wav")
            export_benchmark_variant_ini(
                ini_file,
                tuning,
                variant_id=name,
                description=desc,
                paired_wav=wav_name,
                is_turbo=is_turbo,
                ctx=ini_context,
            )
            dur = audio.size / int(tts.sample_rate)
            on_log(f"  Đã lưu {out_file.name} + {ini_file.name} ({dur:.1f}s)")
            lines.append(
                f"{name}\t{wav_name}\t{ini_name}\t{desc}\t"
                f"{tuning.voice_params_summary(is_turbo=is_turbo)}"
            )
            on_progress(i + 1, total)

        summary_path = bench_root / "summary.txt"
        summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        on_log(f"Đã ghi {summary_path.name}")
        on_log("Mỗi WAV có file .ini cùng tên — dùng «Tải preset» để áp tham số lên GUI.")
        on_done(str(bench_root.resolve()))
    except Exception as e:
        on_error(f"{e}\n{traceback.format_exc()}")
    finally:
        if on_finished is not None:
            on_finished()
