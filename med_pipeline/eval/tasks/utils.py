"""lm-eval `!function` helpers for custom Chinese medical MCQ tasks.

Loaded by the YAMLs in this directory. run_eval.py puts the repo root on PYTHONPATH so these
can reuse the single-source-of-truth MCQ parsers (no duplicated option/letter logic).

Each doc_to_* takes one dataset row (dict) and returns:
  doc_to_text   -> the prompt string ending in the answer cue "答案："
  doc_to_choice -> list of option-letter strings (the classification labels)
  doc_to_target -> 0-based index of the gold letter within doc_to_choice
"""

from __future__ import annotations

from med_pipeline.tools.mcq import normalize_letter, parse_options


def _text(question: str, options) -> str:
    opts = parse_options(options)
    body = "\n".join(f"{l}. {t}" for l, t in opts)
    return f"以下是一道医学单项选择题，请选出正确选项。\n\n{question.strip()}\n{body}\n答案："


def _choice(options) -> list[str]:
    return [l for l, _ in parse_options(options)]


def _target(options, gold_raw) -> int:
    letters = _choice(options)
    gold = normalize_letter(str(gold_raw))
    return letters.index(gold) if gold in letters else 0


# --- MedQA (Chinese) : cols question / options / answer_idx ---
def medqa_doc_to_text(doc):    return _text(doc["question"], doc["options"])
def medqa_doc_to_choice(doc):  return _choice(doc["options"])
def medqa_doc_to_target(doc):  return _target(doc["options"], doc["answer_idx"])


# --- CMExam : cols Question / Options / Answer ---
def cmexam_doc_to_text(doc):    return _text(doc["Question"], doc["Options"])
def cmexam_doc_to_choice(doc):  return _choice(doc["Options"])
def cmexam_doc_to_target(doc):  return _target(doc["Options"], doc["Answer"])
