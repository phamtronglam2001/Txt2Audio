"""GUI Tkinter: chọn TXT → TTS VieNeu Turbo → file WAV hoặc MP3."""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from txt2audio.audio_export import OutputFormat, normalize_output_path, resolve_ffmpeg_executable
from txt2audio.model_paths import (
    AUTO_GGUF_LABEL,
    BUNDLE_SUBDIR_Q8,
    BUNDLE_SUBDIR_TURBO,
    default_models_root,
    ensure_bundle_layout,
    list_gguf_choices,
    parse_gguf_choice,
    resolve_gguf_selection,
    warn_legacy_mixed_dir,
)
from txt2audio.benchmark import run_benchmark_job
from txt2audio.benchmark_dialog import ask_benchmark_plan
from txt2audio.benchmark_plan import BenchmarkPlan
from txt2audio.gguf_loader import vieneu_mode_for_repo
from txt2audio.gui_settings import (
    VOICE_INI_DEFAULT,
    VOICE_INI_REF,
    LoadResult,
    apply_preset_to_gui,
    benchmark_context_from_app,
    default_presets_dir,
    list_preset_ini_files,
    load_preset_file,
    save_gui_preset,
)
from txt2audio.synthesis_tuning import SynthesisTuning
from txt2audio.tts_worker import (
    PrepareModelResult,
    run_prepare_model_job,
    run_synthesis_job,
)

# Giá trị sentinel trong voice_combo_values (tuple thứ 2)
VOICE_PLACEHOLDER = "__PICK__"
VOICE_UNLOADED = "__UNLOADED__"


def main() -> None:
    App(tk.Tk()).run()


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("Txt2Audio — VieNeu Turbo")
        root.minsize(720, 560)

        self.txt_path = tk.StringVar()
        self.out_dir = tk.StringVar(value=str(Path.home() / "Documents"))
        self.out_name = tk.StringVar(value="sach_output.wav")
        self.out_format = tk.StringVar(value="wav")
        self.mp3_bitrate = tk.IntVar(value=32)
        self.ffmpeg_path = tk.StringVar(value="")
        self.model_dir = tk.StringVar(value=str(default_models_root()))
        self.model_file = tk.StringVar(value="")
        self.ref_audio = tk.StringVar()
        self.segment_chars = tk.IntVar(value=8000)
        self.infer_max_chars = tk.IntVar(value=384)
        self.silence_s = tk.DoubleVar(value=0.2)
        self.crossfade_between_s = tk.DoubleVar(value=0.0)
        self.infer_silence_p = tk.DoubleVar(value=0.15)
        self.infer_crossfade_p = tk.DoubleVar(value=0.0)
        self.temperature = tk.DoubleVar(value=1.0)
        self.top_k = tk.IntVar(value=50)
        self.skip_normalize = tk.BooleanVar(value=False)
        self.show_infer_progress = tk.BooleanVar(value=False)
        self.turbo_max_tokens = tk.IntVar(value=1024)
        self.benchmark_sample_chars = tk.IntVar(value=500)
        self.voice_combo_values: list[tuple[str, str | None]] = [
            ("(Chưa nạp model — bấm «Nạp model & giọng»)", VOICE_UNLOADED),
        ]
        self._tts: object | None = None
        self._prepared_gguf: str | None = None
        self._effective_repo_id: str | None = None
        self._tuning_backend: str | None = None
        self._loaded_preset_path: str | None = None
        self._pending_voice_preset_id: str | None = None
        self.preset_pick = tk.StringVar(value="")
        self._bg_thread: threading.Thread | None = None
        self.cancel_event = threading.Event()

        pad = {"padx": 6, "pady": 4}
        r = 0

        ttk.Label(root, text="File TXT (sách):").grid(row=r, column=0, sticky="w", **pad)
        ttk.Entry(root, textvariable=self.txt_path, width=56).grid(row=r, column=1, sticky="ew", **pad)
        ttk.Button(root, text="Chọn…", command=self._browse_txt).grid(row=r, column=2, **pad)
        r += 1

        ttk.Label(root, text="Thư mục lưu file audio:").grid(row=r, column=0, sticky="w", **pad)
        ttk.Entry(root, textvariable=self.out_dir, width=56).grid(row=r, column=1, sticky="ew", **pad)
        ttk.Button(root, text="Chọn…", command=self._browse_out_dir).grid(row=r, column=2, **pad)
        r += 1

        ttk.Label(root, text="Tên file output:").grid(row=r, column=0, sticky="w", **pad)
        ttk.Entry(root, textvariable=self.out_name, width=56).grid(row=r, column=1, sticky="ew", **pad)
        r += 1

        ttk.Label(root, text="Định dạng output:").grid(row=r, column=0, sticky="nw", **pad)
        fmt_box = ttk.Frame(root)
        fmt_box.grid(row=r, column=1, sticky="w", **pad)
        ttk.Radiobutton(
            fmt_box,
            text="WAV (không nén)",
            variable=self.out_format,
            value="wav",
            command=self._on_format_change,
        ).pack(side=tk.LEFT)
        ttk.Radiobutton(
            fmt_box,
            text="MP3 (ffmpeg)",
            variable=self.out_format,
            value="mp3",
            command=self._on_format_change,
        ).pack(side=tk.LEFT, padx=(16, 0))
        r += 1

        ttk.Label(root, text="Chất lượng MP3:").grid(row=r, column=0, sticky="w", **pad)
        mp3_box = ttk.Frame(root)
        mp3_box.grid(row=r, column=1, sticky="w", **pad)
        self.rb_mp3_128 = ttk.Radiobutton(
            mp3_box,
            text="128 kbps — cao (file lớn hơn)",
            variable=self.mp3_bitrate,
            value=128,
        )
        self.rb_mp3_128.pack(side=tk.LEFT)
        self.rb_mp3_32 = ttk.Radiobutton(
            mp3_box,
            text="32 kbps — thấp (tiết kiệm dung lượng)",
            variable=self.mp3_bitrate,
            value=32,
        )
        self.rb_mp3_32.pack(side=tk.LEFT, padx=(16, 0))
        r += 1

        ttk.Label(root, text="Đường dẫn ffmpeg:").grid(row=r, column=0, sticky="w", **pad)
        ff_row = ttk.Frame(root)
        ff_row.grid(row=r, column=1, columnspan=2, sticky="ew", **pad)
        self.ffmpeg_entry = ttk.Entry(ff_row, textvariable=self.ffmpeg_path, width=48)
        self.ffmpeg_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.btn_ffmpeg_browse = ttk.Button(ff_row, text="Chọn…", command=self._browse_ffmpeg)
        self.btn_ffmpeg_browse.pack(side=tk.LEFT, padx=(6, 0))
        ttk.Label(
            ff_row,
            text="(để trống = tìm trong PATH)",
            foreground="#555",
        ).pack(side=tk.LEFT, padx=(8, 0))
        r += 1

        self._mp3_only_widgets: tuple[tk.Widget, ...] = (
            self.rb_mp3_128,
            self.rb_mp3_32,
            self.ffmpeg_entry,
            self.btn_ffmpeg_browse,
        )

        ttk.Label(root, text="Thư mục models (gốc):").grid(row=r, column=0, sticky="w", **pad)
        ttk.Entry(root, textvariable=self.model_dir, width=56).grid(row=r, column=1, sticky="ew", **pad)
        ttk.Button(root, text="Chọn…", command=self._browse_model_dir).grid(row=r, column=2, **pad)
        r += 1

        ttk.Label(
            root,
            text=f"File GGUF ({BUNDLE_SUBDIR_Q8}/ hoặc {BUNDLE_SUBDIR_TURBO}/):",
        ).grid(row=r, column=0, sticky="w", **pad)
        self.model_combo = ttk.Combobox(root, width=53, state="readonly", textvariable=self.model_file)
        self.model_combo.grid(row=r, column=1, sticky="ew", **pad)
        ttk.Button(root, text="Refresh", command=self._refresh_model_files).grid(row=r, column=2, **pad)
        self.model_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_model_file_changed())
        r += 1

        ttk.Label(root, text="Giọng preset:").grid(row=r, column=0, sticky="nw", **pad)
        voice_row = ttk.Frame(root)
        voice_row.grid(row=r, column=1, columnspan=2, sticky="ew", **pad)
        voice_row.columnconfigure(0, weight=1)
        self.voice_combo = ttk.Combobox(
            voice_row,
            width=40,
            state="disabled",
            values=[self.voice_combo_values[0][0]],
        )
        self.voice_combo.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.voice_combo.current(0)
        self.voice_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_voice_selected())
        self.btn_prepare = ttk.Button(
            voice_row,
            text="Nạp model & giọng",
            command=self._prepare_model,
        )
        self.btn_prepare.grid(row=0, column=1, sticky="e")
        r += 1

        ttk.Label(root, text="Giọng clone (WAV/MP3/FLAC, tùy chọn):").grid(row=r, column=0, sticky="w", **pad)
        ttk.Entry(root, textvariable=self.ref_audio, width=46).grid(row=r, column=1, sticky="ew", **pad)
        bf = ttk.Frame(root)
        bf.grid(row=r, column=2, **pad)
        ttk.Button(bf, text="Chọn…", command=self._browse_ref).pack(side=tk.LEFT)
        ttk.Button(bf, text="Xóa", command=lambda: self.ref_audio.set("")).pack(side=tk.LEFT, padx=(4, 0))
        r += 1

        adv = ttk.LabelFrame(root, text="Tuỳ chọn nâng cao (TXT + ghép file)")
        adv.grid(row=r, column=0, columnspan=3, sticky="ew", **pad)
        r += 1
        ap = {"padx": 6, "pady": 3}
        ttk.Label(adv, text="Độ dài đoạn sách (ký tự)").grid(row=0, column=0, sticky="w", **ap)
        ttk.Spinbox(adv, from_=2000, to=20000, increment=500, textvariable=self.segment_chars, width=10).grid(
            row=0, column=1, sticky="w", **ap
        )
        ttk.Label(adv, text="max_chars infer").grid(row=0, column=2, sticky="w", **ap)
        ttk.Spinbox(adv, from_=128, to=1024, increment=32, textvariable=self.infer_max_chars, width=8).grid(
            row=0, column=3, sticky="w", **ap
        )
        ttk.Label(adv, text="Im lặng giữa đoạn sách (s)").grid(row=1, column=0, sticky="w", **ap)
        ttk.Spinbox(adv, from_=0.0, to=2.0, increment=0.05, textvariable=self.silence_s, width=8).grid(
            row=1, column=1, sticky="w", **ap
        )
        ttk.Label(adv, text="Crossfade ghép đoạn (s)").grid(row=1, column=2, sticky="w", **ap)
        ttk.Spinbox(
            adv, from_=0.0, to=1.0, increment=0.05, textvariable=self.crossfade_between_s, width=8
        ).grid(row=1, column=3, sticky="w", **ap)

        inf = ttk.LabelFrame(root, text="Tinh chỉnh giọng đọc (VieNeu infer)")
        inf.grid(row=r, column=0, columnspan=3, sticky="ew", **pad)
        r += 1
        ip = {"padx": 6, "pady": 3}
        ttk.Label(inf, text="temperature").grid(row=0, column=0, sticky="w", **ip)
        ttk.Spinbox(inf, from_=0.1, to=2.0, increment=0.05, textvariable=self.temperature, width=8).grid(
            row=0, column=1, sticky="w", **ip
        )
        ttk.Label(inf, text="(Q8≈1.0, Turbo≈0.4)", foreground="#555").grid(row=0, column=2, sticky="w", **ip)
        ttk.Label(inf, text="top_k").grid(row=0, column=3, sticky="w", **ip)
        ttk.Spinbox(inf, from_=1, to=100, increment=5, textvariable=self.top_k, width=6).grid(
            row=0, column=4, sticky="w", **ip
        )
        ttk.Label(inf, text="Im lặng trong infer (s) [Q8]").grid(row=1, column=0, sticky="w", **ip)
        ttk.Spinbox(inf, from_=0.0, to=1.0, increment=0.05, textvariable=self.infer_silence_p, width=8).grid(
            row=1, column=1, sticky="w", **ip
        )
        ttk.Label(inf, text="Crossfade infer (s) [Q8]").grid(row=1, column=2, sticky="w", **ip)
        ttk.Spinbox(inf, from_=0.0, to=1.0, increment=0.05, textvariable=self.infer_crossfade_p, width=8).grid(
            row=1, column=3, sticky="w", **ip
        )
        ttk.Label(inf, text="max_tokens [Turbo]").grid(row=2, column=0, sticky="w", **ip)
        ttk.Spinbox(inf, from_=256, to=4096, increment=128, textvariable=self.turbo_max_tokens, width=8).grid(
            row=2, column=1, sticky="w", **ip
        )
        ttk.Checkbutton(inf, text="Bỏ chuẩn hóa văn bản (skip_normalize)", variable=self.skip_normalize).grid(
            row=2, column=2, columnspan=2, sticky="w", **ip
        )
        ttk.Checkbutton(inf, text="Thanh tiến độ infer [Turbo]", variable=self.show_infer_progress).grid(
            row=3, column=0, columnspan=2, sticky="w", **ip
        )
        ttk.Button(inf, text="Mặc định Q8", width=12, command=self._tuning_preset_q8).grid(row=3, column=2, **ip)
        ttk.Button(inf, text="Mặc định Turbo", width=12, command=self._tuning_preset_turbo).grid(
            row=3, column=3, **ip
        )
        ttk.Label(inf, text="Mẫu benchmark (ký tự)").grid(row=4, column=0, sticky="w", **ip)
        ttk.Spinbox(inf, from_=80, to=3000, increment=50, textvariable=self.benchmark_sample_chars, width=8).grid(
            row=4, column=1, sticky="w", **ip
        )
        ttk.Label(
            inf,
            text="Benchmark: chọn trục tham số quét (hộp thoại khi bấm nút)",
            foreground="#444",
        ).grid(row=4, column=2, columnspan=2, sticky="w", **ip)

        pre = ttk.LabelFrame(
            root,
            text="Preset cài đặt (.ini) — lưu / tải bộ tham số GUI (nhiều file trong thư mục settings/)",
        )
        pre.grid(row=r, column=0, columnspan=3, sticky="ew", **pad)
        r += 1
        pp = {"padx": 6, "pady": 3}
        ttk.Label(pre, text="File preset:").grid(row=0, column=0, sticky="w", **pp)
        self.preset_combo = ttk.Combobox(pre, width=48, textvariable=self.preset_pick, state="readonly")
        self.preset_combo.grid(row=0, column=1, sticky="ew", **pp)
        pre.columnconfigure(1, weight=1)
        pbtn = ttk.Frame(pre)
        pbtn.grid(row=0, column=2, sticky="e", **pp)
        ttk.Button(pbtn, text="Lưu preset…", command=self._save_gui_preset).pack(side=tk.LEFT)
        ttk.Button(pbtn, text="Tải preset…", command=self._load_gui_preset_dialog).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(pbtn, text="Tải từ danh sách", command=self._load_gui_preset_from_combo).pack(
            side=tk.LEFT, padx=(6, 0)
        )
        ttk.Button(pbtn, text="Làm mới", width=8, command=self._refresh_preset_list).pack(side=tk.LEFT, padx=(6, 0))

        self.progress = ttk.Progressbar(root, mode="determinate", maximum=100)
        self.progress.grid(row=r, column=0, columnspan=2, sticky="ew", **pad)
        btnf = ttk.Frame(root)
        btnf.grid(row=r, column=2, **pad)
        self.btn_start = ttk.Button(btnf, text="Bắt đầu đọc sách", command=self._start, state=tk.DISABLED)
        self.btn_start.pack(side=tk.LEFT)
        self.btn_benchmark = ttk.Button(
            btnf,
            text="Benchmark giọng",
            command=self._start_benchmark,
            state=tk.DISABLED,
        )
        self.btn_benchmark.pack(side=tk.LEFT, padx=(6, 0))
        self.btn_stop = ttk.Button(btnf, text="Dừng", command=self._stop, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=(6, 0))
        r += 1

        logf = ttk.Frame(root)
        logf.grid(row=r, column=0, columnspan=3, sticky="nsew", **pad)
        root.rowconfigure(r, weight=1)
        root.columnconfigure(1, weight=1)
        self.log = tk.Text(logf, height=12, wrap="word", state=tk.DISABLED)
        self.log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(logf, command=self.log.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.log["yscrollcommand"] = sb.set

        foot = (
            "Engine: VieNeu-TTS v2 Turbo (vieneu). Audio có watermark theo model. "
            "https://huggingface.co/pnnbao-ump/VieNeu-TTS-v2-Turbo-GGUF"
        )
        ttk.Label(root, text=foot, wraplength=620, foreground="#444").grid(
            row=r + 1, column=0, columnspan=3, sticky="w", padx=6, pady=4
        )
        self._refresh_model_files()
        self._refresh_preset_list()
        self._apply_tuning_for_gguf_choice(log=False)
        self._on_format_change()
        self.ref_audio.trace_add("write", lambda *_: self.root.after(0, self._update_start_enabled))

    def run(self) -> None:
        self.root.mainloop()

    def _build_tuning(self) -> SynthesisTuning:
        return SynthesisTuning(
            segment_chars=int(self.segment_chars.get()),
            infer_max_chars=int(self.infer_max_chars.get()),
            silence_between_s=float(self.silence_s.get()),
            crossfade_between_s=float(self.crossfade_between_s.get()),
            infer_silence_p=float(self.infer_silence_p.get()),
            infer_crossfade_p=float(self.infer_crossfade_p.get()),
            temperature=float(self.temperature.get()),
            top_k=int(self.top_k.get()),
            skip_normalize=bool(self.skip_normalize.get()),
            show_infer_progress=bool(self.show_infer_progress.get()),
            turbo_max_tokens=int(self.turbo_max_tokens.get()),
        )

    def _apply_tuning_from_dataclass(self, t: SynthesisTuning) -> None:
        self.segment_chars.set(t.segment_chars)
        self.infer_max_chars.set(t.infer_max_chars)
        self.silence_s.set(t.silence_between_s)
        self.crossfade_between_s.set(t.crossfade_between_s)
        self.infer_silence_p.set(t.infer_silence_p)
        self.infer_crossfade_p.set(t.infer_crossfade_p)
        self.temperature.set(t.temperature)
        self.top_k.set(t.top_k)
        self.skip_normalize.set(t.skip_normalize)
        self.show_infer_progress.set(t.show_infer_progress)
        self.turbo_max_tokens.set(t.turbo_max_tokens)

    def _apply_tuning_for_backend(self, backend: str, *, log: bool = True) -> None:
        """backend: «turbo» hoặc «standard» (Q8)."""
        if backend == self._tuning_backend:
            return
        self._tuning_backend = backend
        if backend == "turbo":
            self._apply_tuning_from_dataclass(SynthesisTuning.defaults_turbo())
            label = "Turbo"
        else:
            self._apply_tuning_from_dataclass(SynthesisTuning.defaults_q8())
            label = "Q8 (standard)"
        if log:
            self._append_log(f"Đã áp tham số mặc định Tinh chỉnh cho {label}.")

    def _apply_tuning_for_repo(self, repo_id: str | None, *, log: bool = True) -> None:
        if not repo_id:
            return
        self._apply_tuning_for_backend(vieneu_mode_for_repo(repo_id), log=log)

    def _apply_tuning_for_gguf_choice(self, *, log: bool = False) -> None:
        """Đoán backend từ thư mục vieneu-q8 / vieneu-turbo trong combobox."""
        parsed = parse_gguf_choice(self.model_file.get().strip())
        if parsed is None:
            return
        sub, _name = parsed
        if sub == BUNDLE_SUBDIR_TURBO:
            self._apply_tuning_for_backend("turbo", log=log)
        elif sub == BUNDLE_SUBDIR_Q8:
            self._apply_tuning_for_backend("standard", log=log)

    def _tuning_preset_q8(self) -> None:
        self._tuning_backend = None
        self._apply_tuning_for_backend("standard")

    def _tuning_preset_turbo(self) -> None:
        self._tuning_backend = None
        self._apply_tuning_for_backend("turbo")

    def _refresh_preset_list(self) -> None:
        files = list_preset_ini_files()
        labels = [p.name for p in files]
        self.preset_combo["values"] = labels
        if labels and self.preset_pick.get() not in labels:
            self.preset_pick.set(labels[0])

    def _save_gui_preset(self) -> None:
        initial = default_presets_dir()
        stem = "preset"
        if self._loaded_preset_path:
            stem = Path(self._loaded_preset_path).stem
        path = filedialog.asksaveasfilename(
            title="Lưu preset cài đặt GUI",
            initialdir=str(initial),
            initialfile=f"{stem}.ini",
            defaultextension=".ini",
            filetypes=[("INI preset", "*.ini"), ("All", "*.*")],
        )
        if not path:
            return
        try:
            saved = save_gui_preset(self, Path(path))
            self._loaded_preset_path = str(saved)
            self._refresh_preset_list()
            self.preset_pick.set(saved.name)
            self._append_log(f"Đã lưu preset: {saved}")
            messagebox.showinfo("Txt2Audio", f"Đã lưu preset:\n{saved}")
        except OSError as e:
            messagebox.showerror("Txt2Audio", f"Không lưu được preset:\n{e}")

    def _load_gui_preset_from_path(self, ini_path: Path) -> None:
        try:
            cp = load_preset_file(ini_path)
            result = apply_preset_to_gui(self, cp, source=ini_path)
        except OSError as e:
            messagebox.showerror("Txt2Audio", f"Không đọc được preset:\n{e}")
            return
        self._finish_load_gui_preset(result)

    def _finish_load_gui_preset(self, result: LoadResult) -> None:
        self._refresh_model_files()
        self._reset_model_after_preset_load()
        if not result.had_tuning:
            self._tuning_backend = None
            self._apply_tuning_for_gguf_choice(log=True)
        self._on_format_change()
        self._refresh_preset_list()
        self.preset_pick.set(result.path.name)
        self._append_log(f"Đã tải preset «{result.profile_name}»: {result.path}")
        self._append_log(
            "Bấm «Nạp model & giọng» để nạp GGUF và giọng đã lưu (nếu có trong preset)."
        )
        messagebox.showinfo(
            "Txt2Audio",
            f"Đã tải preset «{result.profile_name}».\n\n"
            "Bấm «Nạp model & giọng» để áp model + giọng preset.",
        )

    def _load_gui_preset_dialog(self) -> None:
        initial = default_presets_dir()
        path = filedialog.askopenfilename(
            title="Tải preset cài đặt GUI",
            initialdir=str(initial),
            filetypes=[("INI preset", "*.ini"), ("All", "*.*")],
        )
        if path:
            self._load_gui_preset_from_path(Path(path))

    def _load_gui_preset_from_combo(self) -> None:
        name = self.preset_pick.get().strip()
        if not name:
            messagebox.showinfo("Txt2Audio", "Chọn một file .ini trong danh sách hoặc dùng «Tải preset…».")
            return
        path = default_presets_dir() / name
        if not path.is_file():
            messagebox.showerror("Txt2Audio", f"Không tìm thấy:\n{path}")
            self._refresh_preset_list()
            return
        self._load_gui_preset_from_path(path)

    def _reset_model_after_preset_load(self) -> None:
        """Sau khi tải preset: bỏ model đang nạp, không đổi tham số Tinh chỉnh."""
        if self._tts is not None:
            try:
                self._tts.close()
            except Exception:
                pass
        self._tts = None
        self._prepared_gguf = None
        self._effective_repo_id = None
        self.voice_combo_values = [("(Tải preset — bấm «Nạp model & giọng»)", VOICE_UNLOADED)]
        self.voice_combo.configure(state="disabled", values=[self.voice_combo_values[0][0]])
        self.voice_combo.current(0)
        self._update_start_enabled()

    def _restore_pending_voice(self) -> None:
        pid = self._pending_voice_preset_id
        if not pid:
            return
        if pid == VOICE_INI_REF:
            self._pending_voice_preset_id = None
            self._update_start_enabled()
            return
        for i, (label, tok) in enumerate(self.voice_combo_values):
            if pid == VOICE_INI_DEFAULT and tok is None and "Mặc định" in label:
                self.voice_combo.current(i)
                self._pending_voice_preset_id = None
                self._update_start_enabled()
                return
            if tok == pid:
                self.voice_combo.current(i)
                self._pending_voice_preset_id = None
                self._update_start_enabled()
                return

    def _on_format_change(self, *_args: object) -> None:
        st = tk.NORMAL if self.out_format.get() == "mp3" else tk.DISABLED
        for w in self._mp3_only_widgets:
            w.configure(state=st)
        self._sync_output_extension()

    def _sync_output_extension(self) -> None:
        name = self.out_name.get().strip()
        if not name:
            return
        p = Path(name)
        stem = p.stem or "output"
        ext = ".mp3" if self.out_format.get() == "mp3" else ".wav"
        if p.suffix.lower() != ext:
            self.out_name.set(f"{stem}{ext}")

    def _browse_txt(self) -> None:
        p = filedialog.askopenfilename(filetypes=[("Text", "*.txt"), ("All", "*.*")])
        if p:
            self.txt_path.set(p)
            stem = Path(p).stem
            ext = ".mp3" if self.out_format.get() == "mp3" else ".wav"
            self.out_name.set(f"{stem}{ext}")
            self.out_dir.set(str(Path(p).parent))

    def _browse_out_dir(self) -> None:
        p = filedialog.askdirectory()
        if p:
            self.out_dir.set(p)

    def _browse_model_dir(self) -> None:
        p = filedialog.askdirectory()
        if p:
            self.model_dir.set(p)
            self._refresh_model_files()

    def _browse_ffmpeg(self) -> None:
        p = filedialog.askopenfilename(
            title="Chọn file thực thi ffmpeg",
            filetypes=[
                ("Executable", "*.exe"),
                ("All files", "*.*"),
            ],
        )
        if p:
            self.ffmpeg_path.set(p)

    def _models_root(self) -> Path:
        return Path(self.model_dir.get().strip()).resolve()

    def _refresh_model_files(self) -> None:
        root = self._models_root()
        ensure_bundle_layout(root)
        warn_legacy_mixed_dir(root, self._append_log)
        values = [AUTO_GGUF_LABEL] + list_gguf_choices(root)
        self.model_combo["values"] = values
        if self.model_file.get() in values:
            return
        self.model_file.set(values[0])

    def _current_gguf_key(self) -> str | None:
        """Khóa «vieneu-q8/file.gguf» đang chọn."""
        resolved = resolve_gguf_selection(self._models_root(), self.model_file.get().strip())
        if resolved is None:
            return None
        _bundle, _path, key = resolved
        return key

    def _on_model_file_changed(self) -> None:
        if self._tts is not None:
            try:
                self._tts.close()
            except Exception:
                pass
        self._tts = None
        self._prepared_gguf = None
        self._effective_repo_id = None
        self._tuning_backend = None
        self.voice_combo_values = [("(Đã đổi file GGUF — bấm «Nạp model & giọng»)", VOICE_UNLOADED)]
        self.voice_combo.configure(state="disabled", values=[self.voice_combo_values[0][0]])
        self.voice_combo.current(0)
        self._apply_tuning_for_gguf_choice(log=True)
        self._update_start_enabled()

    def _browse_ref(self) -> None:
        p = filedialog.askopenfilename(
            filetypes=[
                ("Audio", "*.wav *.mp3 *.flac"),
                ("WAV", "*.wav"),
                ("MP3", "*.mp3"),
                ("FLAC", "*.flac"),
                ("All", "*.*"),
            ]
        )
        if p:
            self.ref_audio.set(p)

    def _append_log(self, line: str) -> None:
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, line + "\n")
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    def _safe_log(self, line: str) -> None:
        self.root.after(0, lambda: self._append_log(line))

    def _set_progress(self, cur: int, total: int) -> None:
        if total <= 0:
            return
        self.progress["value"] = 100.0 * cur / total

    def _safe_progress(self, cur: int, total: int) -> None:
        self.root.after(0, lambda: self._set_progress(cur, total))

    def _on_voice_selected(self) -> None:
        self._update_start_enabled()

    def _update_start_enabled(self) -> None:
        can = self._can_start_synthesis()
        st = tk.NORMAL if can else tk.DISABLED
        self.btn_start.configure(state=st)
        self.btn_benchmark.configure(state=st)

    def _can_start_synthesis(self) -> bool:
        if self._tts is None:
            return False
        current = self._current_gguf_key()
        if not current or self._prepared_gguf != current:
            return False
        if self.ref_audio.get().strip():
            return True
        idx = self.voice_combo.current()
        if idx < 0 or idx >= len(self.voice_combo_values):
            return False
        tok = self.voice_combo_values[idx][1]
        return tok not in (VOICE_PLACEHOLDER, VOICE_UNLOADED)

    def _resolve_local_gguf_selection(self) -> tuple[Path, str] | None:
        """(bundle_dir, basename) — mỗi loại model một cây thư mục riêng."""
        resolved = resolve_gguf_selection(self._models_root(), self.model_file.get().strip())
        if resolved is None:
            messagebox.showerror(
                "Txt2Audio",
                f"Không có file .gguf trong:\n"
                f"  {self._models_root() / BUNDLE_SUBDIR_Q8}\n"
                f"  {self._models_root() / BUNDLE_SUBDIR_TURBO}\n\n"
                f"Đặt Q8 vào {BUNDLE_SUBDIR_Q8}/, Turbo vào {BUNDLE_SUBDIR_TURBO}/ rồi bấm Refresh.",
            )
            return None
        bundle_dir, _gguf_path, key = resolved
        return bundle_dir, key

    def _prepare_model(self) -> None:
        if self._bg_thread is not None and self._bg_thread.is_alive():
            messagebox.showinfo("Txt2Audio", "Đang xử lý — đợi xong hoặc bấm Dừng.")
            return

        sel = self._resolve_local_gguf_selection()
        if sel is None:
            return
        bundle_dir, gguf_key = sel
        _b, gguf_path, _k = resolve_gguf_selection(
            self._models_root(), self.model_file.get().strip()
        ) or (None, None, None)
        assert gguf_path is not None

        if self._tts is not None:
            try:
                self._tts.close()
            except Exception:
                pass
            self._tts = None
            self._prepared_gguf = None

        self.cancel_event.clear()
        self.btn_prepare.configure(state=tk.DISABLED)
        self.btn_start.configure(state=tk.DISABLED)
        self.btn_benchmark.configure(state=tk.DISABLED)
        self.btn_stop.configure(state=tk.NORMAL)
        self._append_log("--- Nạp model & danh sách giọng ---")

        def prep_worker() -> None:
            run_prepare_model_job(
                model_cache_dir=str(bundle_dir),
                preferred_model_filename=gguf_path.name,
                allow_download_if_missing=False,
                cancel_event=self.cancel_event,
                on_log=self._safe_log,
                on_ui=lambda res: self.root.after(0, lambda r=res: self._on_prepare_finished(r)),
            )

        self._bg_thread = threading.Thread(target=prep_worker, daemon=True)
        self._bg_thread.start()

    def _on_prepare_finished(self, res: PrepareModelResult) -> None:
        try:
            if res.error:
                self._append_log(res.error)
                if res.error.strip() != "Đã hủy.":
                    messagebox.showerror("Txt2Audio", res.error[:900])
                self.voice_combo_values = [("(Nạp model thất bại — thử lại)", VOICE_UNLOADED)]
                self.voice_combo.configure(state="disabled", values=[self.voice_combo_values[0][0]])
                self.voice_combo.current(0)
                self._tts = None
                self._prepared_gguf = None
                self._effective_repo_id = None
            else:
                assert res.tts is not None and res.voices is not None
                self._tts = res.tts
                self._prepared_gguf = res.prepared_gguf_filename or self._current_gguf_key()
                self._effective_repo_id = res.effective_repo_id
                if self._tuning_backend != "preset":
                    self._tuning_backend = None
                    self._apply_tuning_for_repo(res.effective_repo_id)
                self.voice_combo_values = [
                    ("(— Chọn giọng đọc —)", VOICE_PLACEHOLDER),
                    ("(Mặc định)", None),
                ]
                for desc, vid in res.voices:
                    self.voice_combo_values.append((f"{desc}  ({vid})", vid))
                self.voice_combo["values"] = [x[0] for x in self.voice_combo_values]
                self.voice_combo.configure(state="readonly")
                self.voice_combo.current(0)
                self._restore_pending_voice()
                if self._pending_voice_preset_id:
                    self._append_log(
                        "Preset: không khớp giọng đã lưu — chọn giọng thủ công trong danh sách."
                    )
                    self._pending_voice_preset_id = None
                self._append_log("Model đã sẵn sàng. Hãy chọn giọng trong danh sách, rồi bấm «Bắt đầu đọc sách».")
        finally:
            self.btn_stop.configure(state=tk.DISABLED)
            self.btn_prepare.configure(state=tk.NORMAL)
            self._bg_thread = None
            self.cancel_event.clear()
            self._update_start_enabled()

    def _selected_preset_id(self) -> str | None:
        idx = self.voice_combo.current()
        if idx < 0 or idx >= len(self.voice_combo_values):
            return None
        tok = self.voice_combo_values[idx][1]
        if tok in (VOICE_PLACEHOLDER, VOICE_UNLOADED):
            return None
        return tok

    def _start(self) -> None:
        tp = self.txt_path.get().strip()
        if not tp or not Path(tp).is_file():
            messagebox.showerror("Txt2Audio", "Chọn file TXT hợp lệ.")
            return
        if self._tts is None:
            messagebox.showerror(
                "Txt2Audio",
                "Bấm «Nạp model & giọng» và đợi tải xong trước khi đọc sách.",
            )
            return
        if not self._can_start_synthesis():
            messagebox.showerror(
                "Txt2Audio",
                "Chọn một giọng trong danh sách (không để mục «— Chọn giọng —»), "
                "hoặc chọn file âm thanh để clone.",
            )
            return

        out_dir = Path(self.out_dir.get().strip())
        fmt: OutputFormat = "mp3" if self.out_format.get() == "mp3" else "wav"
        ffmpeg_bin: str | None = None
        if fmt == "mp3":
            ffmpeg_bin = self.ffmpeg_path.get().strip() or None
            try:
                resolve_ffmpeg_executable(ffmpeg_bin)
            except (RuntimeError, FileNotFoundError) as e:
                messagebox.showerror("Txt2Audio", str(e))
                return

        name = self.out_name.get().strip()
        if not name:
            name = "output.mp3" if fmt == "mp3" else "output.wav"
        out_path = str(normalize_output_path((out_dir / name), fmt).resolve())
        sel = self._resolve_local_gguf_selection()
        if sel is None:
            return
        bundle_dir, _gguf_key = sel
        _b2, gguf_path, _k2 = resolve_gguf_selection(
            self._models_root(), self.model_file.get().strip()
        ) or (None, None, None)
        assert gguf_path is not None

        if self._bg_thread is not None and self._bg_thread.is_alive():
            messagebox.showinfo("Txt2Audio", "Đang xử lý, hãy đợi hoặc bấm Dừng.")
            return

        self.cancel_event.clear()
        self.btn_start.configure(state=tk.DISABLED)
        self.btn_benchmark.configure(state=tk.DISABLED)
        self.btn_prepare.configure(state=tk.DISABLED)
        self.btn_stop.configure(state=tk.NORMAL)
        self.progress["value"] = 0

        ref = self.ref_audio.get().strip() or None
        preset_id = None if ref else self._selected_preset_id()

        def on_done(path: str) -> None:
            def ui() -> None:
                self.progress["value"] = 100
                messagebox.showinfo("Txt2Audio", f"Đã lưu:\n{path}")

            self.root.after(0, ui)

        def on_error(msg: str) -> None:
            def ui() -> None:
                self._append_log(msg)
                messagebox.showerror("Txt2Audio", msg[:800])

            self.root.after(0, ui)

        def job() -> None:
            run_synthesis_job(
                txt_path=tp,
                out_path=out_path,
                tuning=self._build_tuning(),
                model_cache_dir=str(bundle_dir),
                preferred_model_filename=gguf_path.name,
                allow_download_if_missing=False,
                preset_voice_id=preset_id,
                reference_audio_path=ref,
                cancel_event=self.cancel_event,
                tts=self._tts,
                output_format=fmt,
                mp3_bitrate_kbps=int(self.mp3_bitrate.get()),
                ffmpeg_path=ffmpeg_bin,
                on_log=self._safe_log,
                on_progress=self._safe_progress,
                on_model_ready=lambda _v: None,
                on_done=on_done,
                on_error=on_error,
                on_finished=lambda: self.root.after(0, self._reset_work_buttons),
            )

        self._bg_thread = threading.Thread(target=job, daemon=True)
        self._bg_thread.start()
        self._append_log("--- Bắt đầu đọc sách ---")

    def _reset_work_buttons(self) -> None:
        self._bg_thread = None
        self.btn_stop.configure(state=tk.DISABLED)
        self.btn_prepare.configure(state=tk.NORMAL)
        self._update_start_enabled()

    def _start_benchmark(self) -> None:
        tp = self.txt_path.get().strip()
        if not tp or not Path(tp).is_file():
            messagebox.showerror("Txt2Audio", "Chọn file TXT hợp lệ.")
            return
        if self._tts is None or not self._can_start_synthesis():
            messagebox.showerror(
                "Txt2Audio",
                "Nạp model, chọn giọng (hoặc file clone), rồi mới chạy benchmark.",
            )
            return
        if self._effective_repo_id is None:
            messagebox.showerror("Txt2Audio", "Thiếu repo model — nạp lại «Nạp model & giọng».")
            return
        if self._bg_thread is not None and self._bg_thread.is_alive():
            messagebox.showinfo("Txt2Audio", "Đang xử lý, hãy đợi hoặc bấm Dừng.")
            return

        is_turbo = vieneu_mode_for_repo(self._effective_repo_id) == "turbo"
        plan = ask_benchmark_plan(
            self.root,
            is_turbo=is_turbo,
            tuning_base=self._build_tuning(),
        )
        if plan is None:
            return

        out_dir = Path(self.out_dir.get().strip())
        ref = self.ref_audio.get().strip() or None
        preset_id = None if ref else self._selected_preset_id()
        self._run_benchmark_with_plan(plan, txt_path=tp, out_dir=out_dir, ref=ref, preset_id=preset_id)

    def _run_benchmark_with_plan(
        self,
        plan: BenchmarkPlan,
        *,
        txt_path: str,
        out_dir: Path,
        ref: str | None,
        preset_id: str | None,
    ) -> None:
        self.cancel_event.clear()
        self.btn_start.configure(state=tk.DISABLED)
        self.btn_benchmark.configure(state=tk.DISABLED)
        self.btn_prepare.configure(state=tk.DISABLED)
        self.btn_stop.configure(state=tk.NORMAL)
        self.progress["value"] = 0
        self._append_log("--- Benchmark giọng ---")

        def on_done(path: str) -> None:
            def ui() -> None:
                self.progress["value"] = 100
                messagebox.showinfo(
                    "Txt2Audio",
                    f"Đã xuất benchmark WAV + .ini:\n{path}\n\n"
                    "Nghe file ưng ý → «Tải preset» chọn .ini cùng tên.",
                )

            self.root.after(0, ui)

        def on_error(msg: str) -> None:
            def ui() -> None:
                self._append_log(msg)
                messagebox.showerror("Txt2Audio", msg[:800])

            self.root.after(0, ui)

        def job() -> None:
            run_benchmark_job(
                txt_path=txt_path,
                output_dir=out_dir,
                sample_chars=int(self.benchmark_sample_chars.get()),
                tuning_base=self._build_tuning(),
                effective_repo_id=self._effective_repo_id,
                preset_voice_id=preset_id,
                reference_audio_path=ref,
                tts=self._tts,
                cancel_event=self.cancel_event,
                on_log=self._safe_log,
                on_progress=self._safe_progress,
                on_done=on_done,
                on_error=on_error,
                on_finished=lambda: self.root.after(0, self._reset_work_buttons),
                ini_context=benchmark_context_from_app(self),
                plan=plan,
            )

        self._bg_thread = threading.Thread(target=job, daemon=True)
        self._bg_thread.start()

    def _stop(self) -> None:
        self.cancel_event.set()
        self._append_log("(Yêu cầu dừng — sẽ dừng sau đoạn hiện tại)")


if __name__ == "__main__":
    main()
