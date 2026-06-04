# Third-party components & binaries

Txt2Audio **source code** is under the [MIT License](LICENSE) (author: Pham Trong Lam, phamtronglam2001@gmail.com).

This repository is intended to ship **application source only**. Large or separately licensed artifacts must **not** be committed unless you have reviewed the terms below.

---

## What this repo does / does not include

| Item | In Git repo? | Notes |
|------|----------------|-------|
| Python app (`src/txt2audio/`) | Yes | MIT |
| `models/` (`.gguf`, ONNX codec, voices) | **No** (`.gitignore`) | Download locally; see VieNeu model cards |
| `.venv/` | **No** | User runs `uv sync` |
| `ffmpeg` executable | **No** | User installs or points GUI to `ffmpeg.exe` |
| Output WAV/MP3/benchmark | **No** | Generated at runtime |

There are **no** `ffmpeg.exe`, `.gguf`, or `.onnx` files tracked in this project today.

---

## ffmpeg (external, optional — MP3 export)

- **Role:** Txt2Audio invokes `ffmpeg` as an **external program** (`libmp3lame`) when you choose MP3 output. It is **not** bundled in this repo.
- **License:** FFmpeg is typically distributed under **LGPL 2.1+** and/or **GPL 2+**, depending on the build and enabled libraries. See [https://ffmpeg.org/legal.html](https://ffmpeg.org/legal.html).
- **Uploading `ffmpeg.exe` to GitHub:** Possible only if you **comply** with that build’s license (e.g. attribution, license text, source offer for LGPL builds, patent notices). Many projects **do not** commit ffmpeg; they document “install ffmpeg separately.” **Recommended:** do not add `ffmpeg.exe` to this repo; link install instructions in README instead.
- **GPL interaction:** If you distribute a **combined** product that links GPL ffmpeg statically with your app, copyleft may apply to the whole. Txt2Audio only **spawns** ffmpeg as a subprocess for encoding, which is a common pattern; still verify for your distribution scenario.

---

## VieNeu-TTS models & weights (user download)

| Asset | Typical license | In repo? |
|-------|-----------------|----------|
| [VieNeu-TTS-v2-Turbo-GGUF](https://huggingface.co/pnnbao-ump/VieNeu-TTS-v2-Turbo-GGUF) | Apache-2.0 (per model card) | No — place under `models/vieneu-turbo/` locally |
| [VieNeu-TTS-q8-gguf](https://huggingface.co/pnnbao-ump/VieNeu-TTS-q8-gguf) | Apache-2.0 (per model card) | No — place under `models/vieneu-q8/` locally |
| `neucodec-onnx` decoder weights | Follow upstream / HF repo for `neuphonic/neucodec-onnx-decoder-int8` | No — inside `models/.../neucodec-onnx/` |

**GitHub:** Do not push multi‑GB `.gguf` / `.onnx` into git (size limits ~100 MB per file; LFS still has quota). Host models on Hugging Face or release assets separately.

**Audio watermark:** VieNeu outputs may include upstream **AI watermarking** — see model cards.

---

## Python dependencies (installed via `uv sync` / pip)

| Package | Role | Check license |
|---------|------|----------------|
| [vieneu](https://pypi.org/project/vieneu/) | VieNeu-TTS SDK | PyPI metadata / project repo (often Apache-2.0) |
| [neucodec](https://pypi.org/project/neucodec/) | Q8 codec | PyPI metadata / upstream repo |
| `llama-cpp-python` (transitive) | GGUF inference | MIT (common) — verify wheel you install |
| `torch`, etc. (transitive) | ML stack | BSD / other — see installed wheel `LICENSE` in `.venv` |

Re-run license review when you **pin or upgrade** dependencies.

---

## Summary for GitHub upload

1. **Safe to push:** MIT-licensed source, `README`, `LICENSE`, `THIRD_PARTY_NOTICES.md`, `pyproject.toml`, scripts, `.gitignore`.
2. **Do not push by default:** `models/`, `.venv/`, user `settings/*.ini`, benchmark output, audio files.
3. **ffmpeg:** Do not commit binaries unless you intentionally handle LGPL/GPL redistribution; prefer README install steps.
4. **Models:** Keep out of git; document download URLs in README.
