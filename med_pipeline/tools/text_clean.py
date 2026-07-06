"""Text cleaning / quality filtering / hashing for the CPT corpus.

TOOL CONTRACT:
    normalize_whitespace(text)          -> collapsed, control-char-stripped text
    cjk_ratio(text)                     -> fraction of CJK chars in [0,1]
    clean_document(text)                -> normalized doc string
    passes_quality(text, min_chars, min_cjk) -> bool
    doc_hash(text)                      -> sha1 hex of normalized text (for exact dedup)

    Deterministic, no side effects, independently testable. Pure functions.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

_CJK = re.compile(r"[一-鿿]")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_MULTISPACE = re.compile(r"[ \t]{2,}")
_MULTINL = re.compile(r"\n{3,}")


def normalize_whitespace(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _CTRL.sub("", text)
    text = _MULTISPACE.sub(" ", text)
    text = _MULTINL.sub("\n\n", text)
    return text.strip()


def cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    stripped = re.sub(r"\s", "", text)
    if not stripped:
        return 0.0
    return len(_CJK.findall(text)) / len(stripped)


def clean_document(text: str) -> str:
    return normalize_whitespace(text or "")


def passes_quality(text: str, min_chars: int = 200, min_cjk: float = 0.30) -> bool:
    """Keep docs that are long enough and predominantly Chinese (drops nav/boilerplate/EN)."""
    if not text or len(text) < min_chars:
        return False
    return cjk_ratio(text) >= min_cjk


def doc_hash(text: str) -> str:
    return hashlib.sha1(normalize_whitespace(text).encode("utf-8")).hexdigest()
