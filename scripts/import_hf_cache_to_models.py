"""Copy weight đã tải trong HF cache mặc định vào thư mục model portable.

Chạy một lần (không cần activate env nếu dùng .venv python):

    .venv\\Scripts\\python.exe scripts\\import_hf_cache_to_models.py
    .venv\\Scripts\\python.exe scripts\\import_hf_cache_to_models.py --dir models\\vieneu-turbo --repo q8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from txt2audio.gguf_loader import Q8_REPO_ID, TURBO_REPO_ID
from txt2audio.local_assets import ensure_portable_bundle, import_all_known_from_hf_cache
from txt2audio.model_paths import bundle_dir_for_repo, default_models_root


def main() -> None:
    p = argparse.ArgumentParser(description="Import HF hub cache → thư mục model portable (tách q8/turbo)")
    p.add_argument(
        "--root",
        type=Path,
        default=default_models_root(),
        help="Thư mục models gốc (mặc định: ./models)",
    )
    p.add_argument(
        "--repo",
        choices=("turbo", "q8", "both"),
        default="both",
        help="Bộ asset cần import (turbo / q8 / cả hai)",
    )
    args = p.parse_args()
    models_root = args.root.resolve()

    def log(msg: str) -> None:
        print(msg)

    repos: list[str] = []
    if args.repo in ("turbo", "both"):
        repos.append(TURBO_REPO_ID)
    if args.repo in ("q8", "both"):
        repos.append(Q8_REPO_ID)

    for repo_id in repos:
        bundle = bundle_dir_for_repo(repo_id, models_root)
        log(f"\n=== {repo_id} → {bundle} ===")
        import_all_known_from_hf_cache(bundle, repo_id, log)
        try:
            ensure_portable_bundle(bundle, repo_id, allow_download=False, on_log=log)
        except FileNotFoundError as e:
            log(f"(Chưa đủ file offline: {e})")
            log("Chạy app và «Nạp model» (có mạng) hoặc bỏ --no-download.")

    log(f"\nXong. Copy cả thư mục {models_root} (+ .venv) để chạy offline trên máy khác.")


if __name__ == "__main__":
    main()
