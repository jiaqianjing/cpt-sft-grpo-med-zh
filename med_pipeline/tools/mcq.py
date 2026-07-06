"""Multiple-choice answer parsing/normalization for Chinese medical MCQ.

TOOL CONTRACT:
    options_to_letters(n)             -> ["A", "B", ...]           (n in 2..26)
    normalize_letter("（b)")           -> "B" | None
    extract_answer(text, n_options)   -> "A".."?" | None
        Pulls the model's final choice from generated text. Priority:
        1) \boxed{X}   2) 答案/正确答案 …: X   3) last standalone option-letter occurrence.

    Deterministic, no side effects, independently testable. Pure string logic.
"""

from __future__ import annotations

import re
import string

_LETTERS = string.ascii_uppercase


def options_to_letters(n: int) -> list[str]:
    if not 2 <= n <= 26:
        raise ValueError(f"n_options must be in 2..26, got {n}")
    return list(_LETTERS[:n])


def parse_options(options) -> list[tuple[str, str]]:
    """Normalize a source `options` field into [(letter, text), ...], sorted by letter.

    Handles the two schemas seen in the pinned sources:
      - list of {"key"/"label"/"option": <letter>, "value"/"text": <str>}  (MedQA, CMExam)
      - dict {"A": <str>, "B": <str>, ...}                                   (CMB)
    """
    out: list[tuple[str, str]] = []
    if isinstance(options, dict):
        for k, v in options.items():
            letter = normalize_letter(str(k))
            if letter and v:
                out.append((letter, str(v).strip()))
    elif isinstance(options, (list, tuple)):
        for item in options:
            if isinstance(item, dict):
                key = item.get("key") or item.get("label") or item.get("option")
                val = item.get("value") or item.get("text") or item.get("content")
                letter = normalize_letter(str(key)) if key is not None else None
                if letter and val:
                    out.append((letter, str(val).strip()))
    return sorted(out, key=lambda t: t[0])


def render_options(options) -> str:
    """Render options as 'A. text\\nB. text\\n...' for a prompt. Returns '' if unparseable."""
    return "\n".join(f"{letter}. {text}" for letter, text in parse_options(options))


def normalize_letter(s: str | None) -> str | None:
    """Extract a single option letter from a noisy string; supports full-width A-Ｚ."""
    if not s:
        return None
    for ch in s:
        # full-width uppercase Ａ-Ｚ -> ascii
        if "Ａ" <= ch <= "Ｚ":
            return chr(ord(ch) - 0xFEE0)
        if "ａ" <= ch <= "ｚ":  # full-width lowercase
            return chr(ord(ch) - 0xFEE0 - 32)
        if ch.upper() in _LETTERS:
            return ch.upper()
    return None


_BOXED = re.compile(r"\\boxed\{\s*([A-Za-zＡ-Ｚ])\s*\}")
_ANSWER_KV = re.compile(
    r"(?:答案|正确答案|正确选项|选择|answer|final answer)\s*(?:是|为|：|:|应为|应选|选)?\s*"
    r"[（(【\[]?\s*([A-Za-zＡ-Ｚ])",
    re.IGNORECASE,
)
# a single option letter that is NOT part of a longer alphabetic word (so "cat" won't match)
_STANDALONE = re.compile(r"(?<![A-Za-zＡ-Ｚａ-ｚ])[A-Za-zＡ-Ｚ](?![A-Za-zＡ-Ｚａ-ｚ])")


def extract_answer(text: str, n_options: int = 4) -> str | None:
    """Return the model's chosen option letter (uppercased) or None if unparseable."""
    if not text:
        return None
    valid = set(options_to_letters(n_options))

    m = _BOXED.search(text)
    if m:
        letter = normalize_letter(m.group(1))
        if letter in valid:
            return letter

    for m in _ANSWER_KV.finditer(text):
        letter = normalize_letter(m.group(1))
        if letter in valid:
            return letter

    # fallback: last STANDALONE valid option letter (not embedded in a word like "cat")
    for m in reversed(list(_STANDALONE.finditer(text))):
        letter = normalize_letter(m.group(0))
        if letter in valid:
            return letter
    return None
