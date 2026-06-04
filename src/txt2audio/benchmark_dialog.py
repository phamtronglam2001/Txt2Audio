"""Hộp thoại chọn trục tham số trước khi chạy benchmark."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable

from txt2audio.benchmark_plan import (
    CROSSFADE_BETWEEN,
    INFER_CROSSFADE_Q8,
    INFER_MAX_CHARS,
    INFER_SILENCE_Q8,
    SILENCE_BETWEEN,
    TEMPERATURE_Q8,
    TEMPERATURE_TURBO,
    TOP_K_VALUES,
    BenchmarkPlan,
    build_benchmark_variants,
)
from txt2audio.synthesis_tuning import SynthesisTuning


class BenchmarkOptionsDialog(tk.Toplevel):
    def __init__(
        self,
        parent: tk.Misc,
        *,
        is_turbo: bool,
        tuning_base: SynthesisTuning,
        on_confirm: Callable[[BenchmarkPlan], None],
    ) -> None:
        super().__init__(parent)
        self.title("Benchmark — chọn tham số quét")
        self.resizable(True, True)
        self.transient(parent)
        self.grab_set()

        self._is_turbo = is_turbo
        self._tuning_base = tuning_base
        self._on_confirm = on_confirm
        self.result: BenchmarkPlan | None = None

        plan = BenchmarkPlan.default_for_backend(is_turbo)
        self.var_baseline = tk.BooleanVar(value=plan.include_baseline)
        self.var_temp = tk.BooleanVar(value=plan.sweep_temperature)
        self.var_si = tk.BooleanVar(value=plan.sweep_infer_silence and not is_turbo)
        self.var_sb = tk.BooleanVar(value=plan.sweep_silence_between)
        self.var_max = tk.BooleanVar(value=plan.sweep_infer_max_chars)
        self.var_combo = tk.BooleanVar(value=plan.include_natural_combo)
        self.var_topk = tk.BooleanVar(value=plan.sweep_top_k)
        self.var_icf = tk.BooleanVar(value=plan.sweep_infer_crossfade and not is_turbo)
        self.var_bcf = tk.BooleanVar(value=plan.sweep_crossfade_between)

        pad = {"padx": 10, "pady": 6}
        backend = "Turbo" if is_turbo else "Q8 (standard)"
        ttk.Label(
            self,
            text=f"Backend: {backend} — tick các trục cần quét (mỗi giá trị = 1 WAV + .ini)",
            wraplength=480,
        ).grid(row=0, column=0, columnspan=2, sticky="w", **pad)

        box = ttk.LabelFrame(self, text="Trục tham số")
        box.grid(row=1, column=0, columnspan=2, sticky="nsew", **pad)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        rows: list[tuple[tk.BooleanVar, str, str, bool]] = [
            (self.var_baseline, "Baseline", "1 file — đúng giá trị trên panel Tinh chỉnh", True),
            (
                self.var_temp,
                "Temperature",
                f"{len(TEMPERATURE_TURBO if is_turbo else TEMPERATURE_Q8)} mức: "
                f"{TEMPERATURE_TURBO if is_turbo else TEMPERATURE_Q8}",
                True,
            ),
            (
                self.var_si,
                "Im lặng infer (infer_silence_p) [Q8]",
                f"{len(INFER_SILENCE_Q8)} mức — chỉ standard",
                not is_turbo,
            ),
            (
                self.var_sb,
                "Im lặng ghép sách (silence_between_s)",
                f"{len(SILENCE_BETWEEN)} mức: {SILENCE_BETWEEN}",
                True,
            ),
            (
                self.var_max,
                "Độ dài cụm infer (infer_max_chars)",
                f"{len(INFER_MAX_CHARS)} mức: {INFER_MAX_CHARS}",
                True,
            ),
            (self.var_topk, "top_k", f"{len(TOP_K_VALUES)} mức: {TOP_K_VALUES}", True),
            (
                self.var_icf,
                "Crossfade infer [Q8]",
                f"{len(INFER_CROSSFADE_Q8)} mức",
                not is_turbo,
            ),
            (
                self.var_bcf,
                "Crossfade ghép đoạn sách",
                f"{len(CROSSFADE_BETWEEN)} mức",
                True,
            ),
            (self.var_combo, "Combo tự nhiên", "1 file — temp + im lặng dài", True),
        ]

        for i, (var, title, detail, enabled) in enumerate(rows):
            cb = ttk.Checkbutton(box, text=title, variable=var, command=self._update_count)
            cb.grid(row=i, column=0, sticky="w", padx=8, pady=2)
            if not enabled:
                cb.configure(state=tk.DISABLED)
                var.set(False)
            ttk.Label(box, text=detail, foreground="#444", wraplength=420).grid(
                row=i, column=1, sticky="w", padx=4, pady=2
            )

        btn_row = ttk.Frame(self)
        btn_row.grid(row=2, column=0, columnspan=2, sticky="w", **pad)
        ttk.Button(btn_row, text="Chọn tất cả", command=self._select_all).pack(side=tk.LEFT)
        ttk.Button(btn_row, text="Bỏ chọn tất cả", command=self._select_none).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(btn_row, text="Mặc định gợi ý", command=self._select_default).pack(side=tk.LEFT, padx=(8, 0))

        self.lbl_count = ttk.Label(self, text="", font=("", 10, "bold"))
        self.lbl_count.grid(row=3, column=0, columnspan=2, sticky="w", **pad)

        actions = ttk.Frame(self)
        actions.grid(row=4, column=0, columnspan=2, sticky="e", **pad)
        ttk.Button(actions, text="Hủy", command=self._cancel).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(actions, text="Chạy benchmark", command=self._ok).pack(side=tk.RIGHT)

        self.bind("<Escape>", lambda _e: self._cancel())
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self._update_count()
        self.update_idletasks()
        self.geometry("+%d+%d" % (parent.winfo_rootx() + 40, parent.winfo_rooty() + 40))

    def _current_plan(self) -> BenchmarkPlan:
        return BenchmarkPlan(
            include_baseline=self.var_baseline.get(),
            sweep_temperature=self.var_temp.get(),
            sweep_infer_silence=self.var_si.get(),
            sweep_silence_between=self.var_sb.get(),
            sweep_infer_max_chars=self.var_max.get(),
            include_natural_combo=self.var_combo.get(),
            sweep_top_k=self.var_topk.get(),
            sweep_infer_crossfade=self.var_icf.get(),
            sweep_crossfade_between=self.var_bcf.get(),
        )

    def _update_count(self) -> None:
        plan = self._current_plan()
        n = plan.enabled_axis_count(is_turbo=self._is_turbo)
        built = len(
            build_benchmark_variants(
                self._tuning_base, is_turbo=self._is_turbo, plan=plan
            )
        )
        if built == 0:
            self.lbl_count.configure(
                text="Chưa chọn trục nào — tick ít nhất một mục (ví dụ Baseline hoặc Temperature).",
                foreground="#a00",
            )
        else:
            self.lbl_count.configure(
                text=f"Sẽ tạo {built} file WAV + {built} file .ini (ước tính ~{built} lần infer).",
                foreground="#000",
            )

    def _select_all(self) -> None:
        for var, *_ in (
            (self.var_baseline,),
            (self.var_temp,),
            (self.var_sb,),
            (self.var_max,),
            (self.var_combo,),
            (self.var_topk,),
            (self.var_bcf,),
        ):
            var.set(True)
        if not self._is_turbo:
            self.var_si.set(True)
            self.var_icf.set(True)
        self._update_count()

    def _select_none(self) -> None:
        for var in (
            self.var_baseline,
            self.var_temp,
            self.var_si,
            self.var_sb,
            self.var_max,
            self.var_combo,
            self.var_topk,
            self.var_icf,
            self.var_bcf,
        ):
            var.set(False)
        self._update_count()

    def _select_default(self) -> None:
        self._select_none()
        p = BenchmarkPlan.default_for_backend(self._is_turbo)
        self.var_baseline.set(p.include_baseline)
        self.var_temp.set(p.sweep_temperature)
        self.var_si.set(p.sweep_infer_silence)
        self.var_sb.set(p.sweep_silence_between)
        self.var_max.set(p.sweep_infer_max_chars)
        self.var_combo.set(p.include_natural_combo)
        self._update_count()

    def _ok(self) -> None:
        plan = self._current_plan()
        if len(build_benchmark_variants(self._tuning_base, is_turbo=self._is_turbo, plan=plan)) == 0:
            return
        self.result = plan
        self.grab_release()
        self.destroy()
        self._on_confirm(plan)

    def _cancel(self) -> None:
        self.result = None
        self.grab_release()
        self.destroy()


def ask_benchmark_plan(
    parent: tk.Misc,
    *,
    is_turbo: bool,
    tuning_base: SynthesisTuning,
) -> BenchmarkPlan | None:
    """Modal dialog; trả về plan hoặc None nếu hủy."""
    holder: list[BenchmarkPlan | None] = [None]

    def on_confirm(plan: BenchmarkPlan) -> None:
        holder[0] = plan

    dlg = BenchmarkOptionsDialog(
        parent,
        is_turbo=is_turbo,
        tuning_base=tuning_base,
        on_confirm=on_confirm,
    )
    parent.wait_window(dlg)
    return holder[0]
