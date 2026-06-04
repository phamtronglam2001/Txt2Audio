"""Tham số tinh chỉnh TTS — map sang VieNeu infer() và join_audio_chunks."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass
class SynthesisTuning:
    """Giá trị mặc định gần SDK: Q8 standard (temperature=1) / Turbo (temperature=0.4)."""

    @staticmethod
    def defaults_q8() -> SynthesisTuning:
        return SynthesisTuning(
            temperature=1.0,
            top_k=50,
            infer_silence_p=0.2,
            infer_crossfade_p=0.0,
            infer_max_chars=384,
            silence_between_s=0.25,
        )

    @staticmethod
    def defaults_turbo() -> SynthesisTuning:
        return SynthesisTuning(
            temperature=0.4,
            top_k=50,
            infer_silence_p=0.15,
            infer_crossfade_p=0.0,
            infer_max_chars=384,
            silence_between_s=0.2,
            turbo_max_tokens=1024,
            show_infer_progress=False,
        )

    segment_chars: int = 8000
    infer_max_chars: int = 384
    silence_between_s: float = 0.2
    crossfade_between_s: float = 0.0
    infer_silence_p: float = 0.15
    infer_crossfade_p: float = 0.0
    temperature: float = 1.0
    top_k: int = 50
    skip_normalize: bool = False
    show_infer_progress: bool = False
    turbo_max_tokens: int = 1024

    def with_overrides(self, **changes: object) -> SynthesisTuning:
        return replace(self, **changes)

    @staticmethod
    def _fmt_param_num(value: float) -> str:
        """Số gọn cho tên file (0.25 → 0.25, 1.0 → 1)."""
        return format(float(value), "g")

    def voice_params_summary(self, *, is_turbo: bool) -> str:
        """Mô tả tham số ảnh hưởng chất giọng (cho log / summary.txt)."""
        parts = [
            f"temp={self.temperature}",
            f"max_chars={self.infer_max_chars}",
            f"silence_book={self.silence_between_s}s",
        ]
        if not is_turbo:
            parts.append(f"silence_infer={self.infer_silence_p}s")
        return ", ".join(parts)

    def benchmark_wav_stem(self, variant_id: str, *, is_turbo: bool) -> str:
        """
        Tên file WAV (không đuôi): id biến thể + giá trị tham số voice.
        Ví dụ: 02_temp_mid__temp0.4__max384__sb0.2__si0.15
        """
        bits = [
            variant_id,
            f"temp{self._fmt_param_num(self.temperature)}",
            f"max{int(self.infer_max_chars)}",
            f"sb{self._fmt_param_num(self.silence_between_s)}",
        ]
        if not is_turbo:
            bits.append(f"si{self._fmt_param_num(self.infer_silence_p)}")
        stem = "__".join(bits)
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in stem)
        return safe[:180]

    def infer_kwargs(self, *, is_turbo: bool) -> dict:
        """Tham số truyền vào tts.infer() (backend bỏ qua field không hỗ trợ)."""
        kw: dict = {
            "temperature": float(self.temperature),
            "top_k": int(self.top_k),
            "max_chars": int(self.infer_max_chars),
            "skip_normalize": bool(self.skip_normalize),
        }
        if is_turbo:
            kw["show_progress"] = bool(self.show_infer_progress)
            kw["max_tokens"] = int(self.turbo_max_tokens)
        else:
            kw["silence_p"] = float(self.infer_silence_p)
            kw["crossfade_p"] = float(self.infer_crossfade_p)
        return kw
