"""Final-answer extraction and matching, shared by every dataset module.

Follows the protocol the latent-reasoning baselines use (Coconut, CODI): decode
greedily, cut the generation at the first EOS, then compare *only the final answer*
against the gold one after light normalization. A gold answer that merely appears
somewhere in the middle of the generation does not count, and an empty generation is
never correct.
"""
import re
from typing import Optional

# Matches integers and decimals, with an optional sign.
_NUM_RE = re.compile(r"[-+]?\d*\.?\d+")
_WS_RE = re.compile(r"\s+")
# Punctuation stripped before comparing free-form text answers.
_STRIP_CHARS = " \t\n\r.,;:!?\"'`()[]{}<>*_-"


def clean_numeric(text: str) -> str:
    """Drop thousands separators and currency markers that never carry meaning."""
    return text.replace(",", "").replace("$", "").replace("%", "").strip()


def normalize_text(text: str) -> str:
    """Lowercase, collapse whitespace and strip surrounding punctuation."""
    if text is None:
        return ""
    return _WS_RE.sub(" ", text).strip().strip(_STRIP_CHARS).strip().lower()


def strip_answer_prefix(text: str) -> str:
    """Remove a leading 'the answer is' / 'answer:' style lead-in, if present."""
    t = text.strip()
    for pat in (r"^\s*the\s+answer\s+is\s*:?\s*", r"^\s*answer\s*:?\s*", r"^\s*so\s*,?\s*"):
        t = re.sub(pat, "", t, flags=re.IGNORECASE)
    return t


def extract_final_number(text: str) -> Optional[str]:
    """Return the last number in `text`, or the number after '####' when present.

    '####' is the GSM8K answer delimiter; when the model emits it we honour it and
    take the first number that follows, matching the reference datasets.
    """
    cleaned = clean_numeric(text)
    if "####" in cleaned:
        after = cleaned.split("####")[-1]
        matches = _NUM_RE.findall(after)
        if matches:
            return matches[0]
    matches = _NUM_RE.findall(cleaned)
    return matches[-1] if matches else None


def numbers_equal(pred: Optional[str], gold: Optional[str], tol: float = 1e-6) -> bool:
    """Compare two numeric strings, tolerating 42 vs 42.0."""
    if pred is None or gold is None or pred == "" or gold == "":
        return False
    if pred == gold:
        return True
    try:
        return abs(float(pred) - float(gold)) < tol
    except ValueError:
        return False


def match_final_number(prediction: str, target: str) -> bool:
    """Exact-match protocol for numeric tasks: only the final predicted number counts."""
    gold = extract_final_number(target)
    if gold is None:
        gold = clean_numeric(target) or None
    return numbers_equal(extract_final_number(prediction), gold)


def final_content_word(text: str) -> str:
    """Last meaningful word of a normalized answer, e.g. 'sally is a sterpus.' -> 'sterpus'."""
    norm = normalize_text(text)
    if not norm:
        return ""
    return norm.split(" ")[-1].strip(_STRIP_CHARS)


def match_free_form(prediction: str, target: str) -> bool:
    """Exact-match protocol for free-form answers.

    Correct when the normalized strings match outright, or when the final content word
    (the entity a ProsQA-style answer actually decides) matches. Substring containment
    is deliberately NOT accepted: it would score '' and 'is a' as correct against
    'Sally is a sterpus.'.
    """
    pred_norm = normalize_text(strip_answer_prefix(prediction))
    gold_norm = normalize_text(target)
    if not pred_norm or not gold_norm:
        return False
    if pred_norm == gold_norm:
        return True
    gold_word = final_content_word(gold_norm)
    return bool(gold_word) and final_content_word(pred_norm) == gold_word


def match_boolean(prediction: str, target: str) -> bool:
    """True/False questions: the last true/false token the model emits decides."""
    gold = normalize_text(target)
    tokens = re.findall(r"\b(true|false)\b", normalize_text(prediction))
    return bool(tokens) and tokens[-1] == gold
