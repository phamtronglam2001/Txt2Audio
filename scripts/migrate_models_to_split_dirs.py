"""Tách file trong models/vieneu-turbo (lẫn) → models/vieneu-q8 và models/vieneu-turbo.

    .venv\\Scripts\\python.exe scripts\\migrate_models_to_split_dirs.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from txt2audio.gguf_loader import Q8_REPO_ID, TURBO_REPO_ID, repo_id_inferred_from_gguf_filename
from txt2audio.model_paths import (
    BUNDLE_SUBDIR_Q8,
    BUNDLE_SUBDIR_TURBO,
    default_models_root,
    ensure_bundle_layout,
)


def _move_to(src: Path, dest: Path) -> None:
    if not src.exists():
        return
    if dest.resolve() == src.resolve():
        return
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest))
    print(f"  → {dest.relative_to(dest.parent.parent.parent)}")


def main() -> None:
    root = default_models_root()
    ensure_bundle_layout(root)
    turbo_dir = root / BUNDLE_SUBDIR_TURBO
    q8_dir = root / BUNDLE_SUBDIR_Q8

    print(f"Quét {turbo_dir} …")
    for gguf in list(turbo_dir.glob("*.gguf")):
        repo = repo_id_inferred_from_gguf_filename(gguf.name)
        if repo == Q8_REPO_ID:
            print(f"Q8 GGUF: {gguf.name}")
            _move_to(gguf, q8_dir / gguf.name)

    neu = turbo_dir / "neucodec-onnx"
    if neu.is_dir():
        print("neucodec-onnx (chỉ Q8):")
        _move_to(neu, q8_dir / "neucodec-onnx")

    for vf in sorted(turbo_dir.glob("voices*.json")):
        text = vf.read_text(encoding="utf-8")
        if '"Binh"' in text or "'Binh'" in text:
            print(f"voices Q8: {vf.name}")
            dest = q8_dir / "voices.json"
            if not dest.exists():
                shutil.copy2(vf, dest)
                print(f"  → {BUNDLE_SUBDIR_Q8}/voices.json")
        elif "(" in text:
            print(f"voices Turbo: {vf.name}")
            dest = turbo_dir / "voices.json"
            if vf != dest and not dest.exists():
                shutil.copy2(vf, dest)

    print(f"\nCấu trúc:")
    print(f"  {q8_dir}/     — *.gguf Q8, neucodec-onnx/, voices.json")
    print(f"  {turbo_dir}/ — *.gguf Turbo, turbo-codec/, voices.json")


if __name__ == "__main__":
    main()
