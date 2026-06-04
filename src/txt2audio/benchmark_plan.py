"""Kế hoạch benchmark: chọn trục tham số và dải giá trị quét."""

from __future__ import annotations

from dataclasses import dataclass

from txt2audio.synthesis_tuning import SynthesisTuning

Variant = tuple[str, str, SynthesisTuning]

# Dải giá trị mở rộng (mỗi mức = một file WAV + .ini)
TEMPERATURE_TURBO = (0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6)
TEMPERATURE_Q8 = (0.7, 0.75, 0.85, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25)

INFER_SILENCE_Q8 = (0.08, 0.1, 0.15, 0.2, 0.25, 0.28, 0.35)
SILENCE_BETWEEN = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4)
INFER_MAX_CHARS = (192, 256, 320, 384, 448, 512, 640)

TOP_K_VALUES = (20, 35, 50, 65, 80)
INFER_CROSSFADE_Q8 = (0.0, 0.05, 0.1, 0.15, 0.2)
CROSSFADE_BETWEEN = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25)


@dataclass
class BenchmarkPlan:
    """Trục nào được quét khi benchmark (mỗi giá trị = một biến thể)."""

    include_baseline: bool = True
    sweep_temperature: bool = True
    sweep_infer_silence: bool = True
    sweep_silence_between: bool = True
    sweep_infer_max_chars: bool = True
    include_natural_combo: bool = True
    sweep_top_k: bool = False
    sweep_infer_crossfade: bool = False
    sweep_crossfade_between: bool = False

    @staticmethod
    def default_for_backend(is_turbo: bool) -> BenchmarkPlan:
        p = BenchmarkPlan()
        if is_turbo:
            p.sweep_infer_silence = False
            p.sweep_infer_crossfade = False
        return p

    def enabled_axis_count(self, *, is_turbo: bool) -> int:
        n = 0
        if self.include_baseline:
            n += 1
        if self.sweep_temperature:
            n += len(TEMPERATURE_TURBO if is_turbo else TEMPERATURE_Q8)
        if self.sweep_infer_silence and not is_turbo:
            n += len(INFER_SILENCE_Q8)
        if self.sweep_silence_between:
            n += len(SILENCE_BETWEEN)
        if self.sweep_infer_max_chars:
            n += len(INFER_MAX_CHARS)
        if self.include_natural_combo:
            n += 1
        if self.sweep_top_k:
            n += len(TOP_K_VALUES)
        if self.sweep_infer_crossfade and not is_turbo:
            n += len(INFER_CROSSFADE_Q8)
        if self.sweep_crossfade_between:
            n += len(CROSSFADE_BETWEEN)
        return n


def _slug_value(v: float | int) -> str:
    if isinstance(v, float):
        return format(v, "g").replace(".", "p")
    return str(v)


def build_benchmark_variants(
    base: SynthesisTuning,
    *,
    is_turbo: bool,
    plan: BenchmarkPlan,
) -> list[Variant]:
    out: list[Variant] = []
    seq = 0

    def add(name_part: str, desc: str, **kw: object) -> None:
        nonlocal seq
        name = f"{seq:02d}_{name_part}"
        seq += 1
        out.append((name, desc, base.with_overrides(**kw)))

    if plan.include_baseline:
        add("baseline", "Theo panel Tinh chỉnh hiện tại")

    if plan.sweep_temperature:
        for t in (TEMPERATURE_TURBO if is_turbo else TEMPERATURE_Q8):
            add(f"temp_{_slug_value(t)}", f"temperature = {t}", temperature=t)

    if plan.sweep_infer_silence and not is_turbo:
        for s in INFER_SILENCE_Q8:
            add(f"si_{_slug_value(s)}", f"infer_silence_p = {s}s", infer_silence_p=s)

    if plan.sweep_silence_between:
        for s in SILENCE_BETWEEN:
            add(f"sb_{_slug_value(s)}", f"silence_between_s = {s}s", silence_between_s=s)

    if plan.sweep_infer_max_chars:
        for m in INFER_MAX_CHARS:
            add(f"max_{m}", f"infer_max_chars = {m}", infer_max_chars=m)

    if plan.sweep_top_k:
        for k in TOP_K_VALUES:
            add(f"topk_{k}", f"top_k = {k}", top_k=k)

    if plan.sweep_infer_crossfade and not is_turbo:
        for c in INFER_CROSSFADE_Q8:
            add(f"icf_{_slug_value(c)}", f"infer_crossfade_p = {c}s", infer_crossfade_p=c)

    if plan.sweep_crossfade_between:
        for c in CROSSFADE_BETWEEN:
            add(f"bcf_{_slug_value(c)}", f"crossfade_between_s = {c}s", crossfade_between_s=c)

    if plan.include_natural_combo:
        temp_natural = 0.4 if is_turbo else 0.9
        add(
            "natural_combo",
            "combo: temp + im lặng dài",
            temperature=temp_natural,
            infer_silence_p=0.25,
            silence_between_s=0.3,
        )

    return out


# Giữ tương thích code cũ
def voice_benchmark_variants(
    base: SynthesisTuning,
    *,
    is_turbo: bool,
) -> list[Variant]:
    return build_benchmark_variants(
        base, is_turbo=is_turbo, plan=BenchmarkPlan.default_for_backend(is_turbo)
    )
