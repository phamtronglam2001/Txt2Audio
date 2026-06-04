"""Hai cây thư mục offline tách riêng: Q8 (standard) và Turbo."""

from __future__ import annotations

from pathlib import Path

from txt2audio.gguf_loader import (
    Q8_REPO_ID,
    TURBO_REPO_ID,
    repo_id_inferred_from_gguf_filename,
    resolve_backbone_repo_for_gguf,
)

# Thư mục con dưới models/ — mỗi loại một cây, không trộn codec/voices.
BUNDLE_SUBDIR_Q8 = "vieneu-q8"
BUNDLE_SUBDIR_TURBO = "vieneu-turbo"

BUNDLE_SUBDIR_BY_REPO: dict[str, str] = {
    Q8_REPO_ID: BUNDLE_SUBDIR_Q8,
    TURBO_REPO_ID: BUNDLE_SUBDIR_TURBO,
}

AUTO_GGUF_LABEL = "(Tự động: file đầu tiên tìm thấy)"


def default_models_root() -> Path:
    return (Path.cwd() / "models").resolve()


def bundle_dir_for_repo(backbone_repo_id: str, models_root: Path | None = None) -> Path:
    root = (models_root or default_models_root()).resolve()
    sub = BUNDLE_SUBDIR_BY_REPO.get(backbone_repo_id, BUNDLE_SUBDIR_TURBO)
    path = root / sub
    path.mkdir(parents=True, exist_ok=True)
    return path


def bundle_dir_for_gguf_file(gguf_path: Path, models_root: Path | None = None) -> Path:
    repo_id = resolve_backbone_repo_for_gguf(gguf_path.resolve())
    return bundle_dir_for_repo(repo_id, models_root)


def ensure_bundle_layout(models_root: Path | None = None) -> None:
    """Tạo sẵn cả hai cây thư mục."""
    root = (models_root or default_models_root()).resolve()
    root.mkdir(parents=True, exist_ok=True)
    for sub in (BUNDLE_SUBDIR_Q8, BUNDLE_SUBDIR_TURBO):
        (root / sub).mkdir(parents=True, exist_ok=True)


def list_gguf_choices(models_root: Path) -> list[str]:
    """Danh sách lựa chọn combobox: «vieneu-q8/foo.gguf»."""
    ensure_bundle_layout(models_root)
    out: list[str] = []
    for sub in (BUNDLE_SUBDIR_Q8, BUNDLE_SUBDIR_TURBO):
        d = models_root / sub
        if not d.is_dir():
            continue
        for g in sorted(d.glob("*.gguf"), key=lambda p: p.name.lower()):
            out.append(f"{sub}/{g.name}")
    return out


def parse_gguf_choice(choice: str) -> tuple[str, str] | None:
    """«vieneu-q8/a.gguf» → (subdir, filename)."""
    c = choice.strip().replace("\\", "/")
    if not c or c == AUTO_GGUF_LABEL:
        return None
    if "/" not in c:
        return None
    sub, name = c.split("/", 1)
    if sub not in (BUNDLE_SUBDIR_Q8, BUNDLE_SUBDIR_TURBO) or not name.lower().endswith(".gguf"):
        return None
    return sub, name


def resolve_gguf_selection(
    models_root: Path,
    choice: str,
) -> tuple[Path, Path, str] | None:
    """
    Trả về (bundle_dir, đường_dẫn_gguf, khóa_lưu_prepared).

    Khóa prepared dạng «vieneu-q8/file.gguf».
    """
    models_root = models_root.resolve()
    ensure_bundle_layout(models_root)

    parsed = parse_gguf_choice(choice)
    if parsed:
        sub, filename = parsed
        bundle = models_root / sub
        gguf_path = bundle / filename
        if not gguf_path.is_file():
            return None
        return bundle, gguf_path, f"{sub}/{filename}"

    # Tự động: q8 trước, rồi turbo
    for sub in (BUNDLE_SUBDIR_Q8, BUNDLE_SUBDIR_TURBO):
        d = models_root / sub
        if not d.is_dir():
            continue
        ggufs = sorted(d.glob("*.gguf"), key=lambda p: p.name.lower())
        if ggufs:
            g = ggufs[0]
            return d, g, f"{sub}/{g.name}"
    return None


def warn_legacy_mixed_dir(models_root: Path, on_log) -> None:
    """Cảnh báo nếu vẫn còn .gguf / voices ngay trong thư mục gốc models/ (cấu trúc cũ)."""
    root = models_root.resolve()
    legacy_gguf = list(root.glob("*.gguf"))
    if legacy_gguf:
        on_log(
            f"⚠ Có {len(legacy_gguf)} file .gguf trong {root} — "
            f"hãy chuyển vào {BUNDLE_SUBDIR_Q8}/ hoặc {BUNDLE_SUBDIR_TURBO}/."
        )
