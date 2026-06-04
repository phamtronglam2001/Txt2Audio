"""Tải / cache mọi weight TTS vào một thư mục portable (copy sang máy khác được)."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Callable, Optional

from huggingface_hub import hf_hub_download, try_to_load_from_cache

from txt2audio.gguf_loader import Q8_REPO_ID, TURBO_REPO_ID, vieneu_mode_for_repo

LogFn = Callable[[str], None]

VOICES_JSON = "voices.json"
NEUCODEC_ONNX_DIR = "neucodec-onnx"
TURBO_CODEC_DIR = "turbo-codec"

NEUCODEC_REPO = "neuphonic/neucodec-onnx-decoder-int8"
TURBO_CODEC_REPO = "pnnbao-ump/VieNeu-Codec"
TURBO_DECODER_FILE = "vieneu_decoder.onnx"
TURBO_ENCODER_FILE = "vieneu_encoder.onnx"
NEUCODEC_ONNX_FILE = "model.onnx"
NEUCODEC_META_FILE = "meta.yaml"


def voices_json_for_repo(bundle_dir: Path, backbone_repo_id: str) -> Path:
    """Mỗi repo HF một file voices riêng — tránh lẫn Turbo/Q8 trong cùng thư mục."""
    slug = backbone_repo_id.replace("/", "__")
    return bundle_dir / f"voices__{slug}.json"


def _voices_json_matches_repo(dest: Path, backbone_repo_id: str) -> bool:
    if not dest.is_file():
        return False
    try:
        presets = list(json.loads(dest.read_text(encoding="utf-8")).get("presets", {}).keys())
    except (OSError, json.JSONDecodeError):
        return False
    if not presets:
        return False
    if backbone_repo_id == Q8_REPO_ID:
        return any(k in presets for k in ("Binh", "Tuyen", "Vinh", "Doan", "Ly", "Ngoc"))
    if backbone_repo_id == TURBO_REPO_ID:
        return any("(" in k for k in presets)
    return True


def publish_voices_json(bundle_dir: Path, backbone_repo_id: str, on_log: LogFn | None = None) -> Path:
    """Copy voices đúng repo → voices.json (SDK đọc file này cạnh GGUF)."""
    src = voices_json_for_repo(bundle_dir, backbone_repo_id)
    dest = bundle_dir / VOICES_JSON
    if not src.is_file():
        raise FileNotFoundError(f"Thiếu {src.name} cho repo {backbone_repo_id}")
    shutil.copy2(src, dest)
    if on_log:
        on_log(f"Dùng voices.json từ {src.name}")
    return dest


def bundle_dir_from_gguf(gguf_path: Path, models_root: Path | None = None) -> Path:
    """Thư mục bundle đúng loại (q8 / turbo), không lẫn config."""
    from txt2audio.model_paths import bundle_dir_for_gguf_file

    return bundle_dir_for_gguf_file(gguf_path, models_root)


def default_hf_hub_cache() -> Path:
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _copy_if_newer(src: Path, dest: Path) -> bool:
    if not src.is_file():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size == src.stat().st_size:
        return True
    shutil.copy2(src, dest)
    return True


def _import_from_hf_cache(repo_id: str, filename: str, dest: Path) -> bool:
    cached = try_to_load_from_cache(repo_id=repo_id, filename=filename, repo_type="model")
    if not cached or not isinstance(cached, str):
        return False
    return _copy_if_newer(Path(cached), dest)


def _scan_hub_snapshots_for_file(filename: str, dest: Path, on_log: LogFn) -> bool:
    """Dò file trong cache HF mặc định (khi try_to_load_from_cache không có)."""
    hub = default_hf_hub_cache()
    if not hub.is_dir():
        return False
    needle = filename.lower()
    best: Path | None = None
    for snap_dir in hub.glob("models--*/snapshots/*"):
        candidate = snap_dir / filename
        if candidate.is_file():
            if best is None or candidate.stat().st_mtime > best.stat().st_mtime:
                best = candidate
    if best is None:
        return False
    on_log(f"Copy từ HF cache → {dest.relative_to(dest.parent.parent)}: {best.name}")
    return _copy_if_newer(best, dest)


def ensure_local_file(
    *,
    repo_id: str,
    filename: str,
    dest: Path,
    allow_download: bool,
    on_log: LogFn,
) -> Path:
    """Đảm bảo file nằm tại dest (import cache HF hoặc tải vào thư mục đó)."""
    if dest.is_file():
        return dest

    if _import_from_hf_cache(repo_id, filename, dest):
        on_log(f"Đã copy local: {dest.name}")
        return dest

    if _scan_hub_snapshots_for_file(filename, dest, on_log):
        return dest

    if not allow_download:
        raise FileNotFoundError(f"Thiếu file local '{dest}' và không cho phép tải.")

    on_log(f"Đang tải '{filename}' từ '{repo_id}' → {dest.parent.name}/ …")
    dl = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(dest.parent),
    )
    return Path(dl)


def ensure_voices_json(
    bundle_dir: Path,
    backbone_repo_id: str,
    *,
    allow_download: bool,
    on_log: LogFn,
) -> Path:
    dest = voices_json_for_repo(bundle_dir, backbone_repo_id)
    canonical = bundle_dir / VOICES_JSON
    if dest.is_file() and not _voices_json_matches_repo(dest, backbone_repo_id):
        on_log(f"Xóa {dest.name} không khớp repo — tải lại.")
        dest.unlink(missing_ok=True)
    if canonical.is_file() and not _voices_json_matches_repo(canonical, backbone_repo_id):
        on_log("Xóa voices.json cũ (sai loại model Turbo/Q8).")
        canonical.unlink(missing_ok=True)
    out = ensure_local_file(
        repo_id=backbone_repo_id,
        filename=VOICES_JSON,
        dest=dest,
        allow_download=allow_download,
        on_log=on_log,
    )
    publish_voices_json(bundle_dir, backbone_repo_id, on_log)
    return out


def ensure_neucodec_onnx(
    bundle_dir: Path,
    *,
    allow_download: bool,
    on_log: LogFn,
) -> Path:
    onnx_dir = bundle_dir / NEUCODEC_ONNX_DIR
    onnx_dir.mkdir(parents=True, exist_ok=True)
    ensure_local_file(
        repo_id=NEUCODEC_REPO,
        filename=NEUCODEC_ONNX_FILE,
        dest=onnx_dir / NEUCODEC_ONNX_FILE,
        allow_download=allow_download,
        on_log=on_log,
    )
    try:
        ensure_local_file(
            repo_id=NEUCODEC_REPO,
            filename=NEUCODEC_META_FILE,
            dest=onnx_dir / NEUCODEC_META_FILE,
            allow_download=allow_download,
            on_log=on_log,
        )
    except FileNotFoundError:
        pass
    return onnx_dir / NEUCODEC_ONNX_FILE


def ensure_turbo_codec(
    bundle_dir: Path,
    *,
    allow_download: bool,
    on_log: LogFn,
) -> tuple[Path, Path]:
    codec_dir = bundle_dir / TURBO_CODEC_DIR
    codec_dir.mkdir(parents=True, exist_ok=True)
    decoder = ensure_local_file(
        repo_id=TURBO_CODEC_REPO,
        filename=TURBO_DECODER_FILE,
        dest=codec_dir / TURBO_DECODER_FILE,
        allow_download=allow_download,
        on_log=on_log,
    )
    encoder = ensure_local_file(
        repo_id=TURBO_CODEC_REPO,
        filename=TURBO_ENCODER_FILE,
        dest=codec_dir / TURBO_ENCODER_FILE,
        allow_download=allow_download,
        on_log=on_log,
    )
    return decoder, encoder


def ensure_portable_bundle(
    bundle_dir: Path,
    backbone_repo_id: str,
    *,
    allow_download: bool,
    on_log: LogFn,
) -> None:
    """Chuẩn bị đủ file trong bundle_dir cho repo đang chọn (offline / copy máy khác)."""
    bundle_dir.mkdir(parents=True, exist_ok=True)
    import_all_known_from_hf_cache(bundle_dir, backbone_repo_id, on_log)
    on_log(f"Thư mục model portable: {bundle_dir}")
    ensure_voices_json(bundle_dir, backbone_repo_id, allow_download=allow_download, on_log=on_log)
    mode = vieneu_mode_for_repo(backbone_repo_id)
    if mode == "standard":
        ensure_neucodec_onnx(bundle_dir, allow_download=allow_download, on_log=on_log)
    else:
        ensure_turbo_codec(bundle_dir, allow_download=allow_download, on_log=on_log)


def import_gguf_from_hf_cache(bundle_dir: Path, backbone_repo_id: str, on_log: LogFn) -> None:
    """Copy file .gguf từ HF hub cache vào thư mục bundle (cùng cấp voices.json)."""
    hub = default_hf_hub_cache()
    if not hub.is_dir():
        return
    repo_slug = "models--" + backbone_repo_id.replace("/", "--")
    repo_cache = hub / repo_slug / "snapshots"
    if not repo_cache.is_dir():
        return
    for gguf in repo_cache.glob("*/*.gguf"):
        dest = bundle_dir / gguf.name
        if dest.is_file():
            continue
        on_log(f"Copy GGUF từ cache → {dest.name}")
        _copy_if_newer(gguf, dest)


def import_all_known_from_hf_cache(bundle_dir: Path, backbone_repo_id: str, on_log: LogFn) -> None:
    """Một lần: kéo file đã tải trong HF cache vào thư mục bundle (không cần mạng)."""
    bundle_dir.mkdir(parents=True, exist_ok=True)
    on_log("Đang import file từ Hugging Face cache (nếu có)…")
    import_gguf_from_hf_cache(bundle_dir, backbone_repo_id, on_log)
    for repo_id, fname, dest in (
        (backbone_repo_id, VOICES_JSON, voices_json_for_repo(bundle_dir, backbone_repo_id)),
        (NEUCODEC_REPO, NEUCODEC_ONNX_FILE, bundle_dir / NEUCODEC_ONNX_DIR / NEUCODEC_ONNX_FILE),
        (NEUCODEC_REPO, NEUCODEC_META_FILE, bundle_dir / NEUCODEC_ONNX_DIR / NEUCODEC_META_FILE),
        (TURBO_CODEC_REPO, TURBO_DECODER_FILE, bundle_dir / TURBO_CODEC_DIR / TURBO_DECODER_FILE),
        (TURBO_CODEC_REPO, TURBO_ENCODER_FILE, bundle_dir / TURBO_CODEC_DIR / TURBO_ENCODER_FILE),
    ):
        if dest.is_file():
            continue
        if _import_from_hf_cache(repo_id, fname, dest):
            on_log(f"  + {dest.relative_to(bundle_dir)}")
        elif _scan_hub_snapshots_for_file(fname, dest, on_log):
            pass
