"""GGUF loader helpers for multiple Hugging Face repos."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from huggingface_hub import hf_hub_download, list_repo_files

LogFn = Callable[[str], None]

TURBO_REPO_ID = "pnnbao-ump/VieNeu-TTS-v2-Turbo-GGUF"
Q8_REPO_ID = "pnnbao-ump/VieNeu-TTS-q8-gguf"

DEFAULT_GGUF_BY_REPO: dict[str, str] = {
    TURBO_REPO_ID: "vieneu-tts-v2-turbo.gguf",
}

# VieNeu SDK backend mode mapping by GGUF family.
# - Turbo v2 GGUF uses turbo backend.
# - Q8 GGUF repo is from standard VieNeu-TTS line.
VIENEU_MODE_BY_REPO: dict[str, str] = {
    TURBO_REPO_ID: "turbo",
    Q8_REPO_ID: "standard",
}


def vieneu_mode_for_repo(repo_id: str) -> str:
    """Return SDK mode for a selected GGUF repo."""
    return VIENEU_MODE_BY_REPO.get(repo_id, "turbo")


def repo_id_inferred_from_gguf_filename(filename: str) -> str | None:
    """Đoán repo HF từ tên file .gguf (Q8 vs Turbo)."""
    low = filename.lower()
    if "turbo" in low:
        return TURBO_REPO_ID
    if any(tag in low for tag in ("q8", "q4", "q6", "q5", "q3")):
        return Q8_REPO_ID
    return None


def resolve_backbone_repo_for_gguf(
    model_path: Path,
    on_log: LogFn | None = None,
) -> str:
    """Chọn repo HF / backend SDK chỉ từ tên file .gguf."""
    inferred = repo_id_inferred_from_gguf_filename(model_path.name)
    if inferred is not None:
        mode = vieneu_mode_for_repo(inferred)
        if on_log:
            on_log(f"Nhận dạng «{model_path.name}» → mode='{mode}'")
        return inferred
    if on_log:
        on_log(f"Không nhận dạng loại từ «{model_path.name}», mặc định Turbo.")
    return TURBO_REPO_ID


def choose_remote_gguf_filename(repo_id: str) -> str:
    """Pick a GGUF filename from repo (priority: known default -> q8-ish -> first)."""
    known = DEFAULT_GGUF_BY_REPO.get(repo_id)
    if known:
        return known

    files = list_repo_files(repo_id=repo_id)
    ggufs = sorted([f for f in files if f.lower().endswith(".gguf")], key=str.lower)
    if not ggufs:
        raise FileNotFoundError(f"Repo '{repo_id}' không có file .gguf")

    # Prefer Q8-like names for q8 repos.
    for f in ggufs:
        low = f.lower()
        if "q8" in low or "q8_0" in low:
            return f
    return ggufs[0]


def resolve_gguf_model_path(
    *,
    repo_id: str,
    model_cache_dir: str,
    preferred_model_filename: Optional[str],
    allow_download_if_missing: bool,
    on_log: LogFn,
) -> Path:
    cache_dir = Path(model_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if preferred_model_filename:
        preferred = cache_dir / preferred_model_filename
        if preferred.is_file():
            on_log(f"Dùng GGUF local: {preferred.name}")
            return preferred
        if not allow_download_if_missing:
            raise FileNotFoundError(
                f"Không tìm thấy GGUF '{preferred_model_filename}' trong {cache_dir}."
            )
        on_log(f"Đang tải GGUF '{preferred_model_filename}' từ '{repo_id}' vào {cache_dir} ...")
        dl_path = hf_hub_download(
            repo_id=repo_id,
            filename=preferred_model_filename,
            local_dir=str(cache_dir),
        )
        return Path(dl_path)

    local_models = sorted(cache_dir.glob("*.gguf"), key=lambda p: p.name.lower())
    if local_models:
        on_log(f"Dùng GGUF local: {local_models[0].name}")
        return local_models[0]

    if not allow_download_if_missing:
        raise FileNotFoundError(
            f"Không có file .gguf trong {cache_dir}. Hãy chọn model hoặc cho phép tải tự động."
        )

    remote_filename = choose_remote_gguf_filename(repo_id)
    on_log(f"Đang tải GGUF mặc định '{remote_filename}' từ '{repo_id}' vào {cache_dir} ...")
    dl_path = hf_hub_download(
        repo_id=repo_id,
        filename=remote_filename,
        local_dir=str(cache_dir),
    )
    return Path(dl_path)
