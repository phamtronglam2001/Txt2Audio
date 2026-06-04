"""Chia văn bản sách thành các đoạn trước khi gọi TTS (tiến độ GUI + kiểm soát bộ nhớ)."""

from __future__ import annotations

import re
from typing import List

from vieneu_utils.core_utils import split_text_into_chunks


def normalize_book_text(raw: str) -> str:
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    raw = re.sub(r"[ \t]+", " ", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return raw.strip()


def book_segments(raw: str, max_segment_chars: int) -> List[str]:
    """Chia sách thành các đoạn văn bản thuần (ký tự), mỗi đoạn <= max_segment_chars."""
    text = normalize_book_text(raw)
    if not text:
        return []
    return split_text_into_chunks(text, max_chars=max_segment_chars)


def read_txt_file(path: str, encodings: tuple[str, ...] = ("utf-8", "utf-8-sig")) -> str:
    last_err: Exception | None = None
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError as e:
            last_err = e
            continue
    raise ValueError(f"Không đọc được file với UTF-8/utf-8-sig: {last_err}")
