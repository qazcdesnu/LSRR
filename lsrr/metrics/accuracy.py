"""최종답 exact match 프로토콜 (제안서 §6).

잠재 추론 베이스라인(ICoT·Coconut·CODI)이 공유하는 규약을 따른다: **greedy 디코딩,
첫 EOS에서 절단, 정규화 후 최종 답만 비교.** 생성 중간에 정답이 등장했다는 이유로
정답 처리하지 않는다 — 그렇게 하면 정확도가 부풀고 외부 공표치와 비교가 깨진다.

개작: Legacy_LSRR/lsrr/data/answer_scoring.py
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

_NUM = re.compile(r"[-+]?\d*\.?\d+")
_WS = re.compile(r"\s+")
_STRIP = " \t\n\r.,;:!?\"'`()[]{}<>*_-"


def normalize_text(text: str) -> str:
    """소문자화·공백 정규화·둘러싼 문장부호 제거."""
    if not text:
        return ""
    return _WS.sub(" ", text).strip().strip(_STRIP).strip().lower()


def clean_numeric(text: str) -> str:
    """자릿수 구분자와 통화 기호 제거."""
    return text.replace(",", "").replace("$", "").replace("%", "").strip()


def extract_final_number(text: str) -> Optional[str]:
    """마지막 숫자. `####` 구분자가 있으면 그 뒤 첫 숫자."""
    cleaned = clean_numeric(text)
    if "####" in cleaned:
        after = _NUM.findall(cleaned.split("####")[-1])
        if after:
            return after[0]
    found = _NUM.findall(cleaned)
    return found[-1] if found else None


def numbers_equal(pred: Optional[str], gold: Optional[str], tol: float = 1e-6) -> bool:
    """42 와 42.0 을 같게 본다."""
    if not pred or not gold:
        return False
    if pred == gold:
        return True
    try:
        return abs(float(pred) - float(gold)) < tol
    except ValueError:
        return False


def match_final_number(prediction: str, target: str) -> bool:
    """수치 과제: 예측의 **마지막** 숫자만 채점 대상이다."""
    gold = extract_final_number(target) or (clean_numeric(target) or None)
    return numbers_equal(extract_final_number(prediction), gold)


def match_free_form(prediction: str, target: str) -> bool:
    """자유 형식: 정규화 완전일치, 또는 최종 개체어 일치.

    부분 문자열 포함은 받지 않는다 — 빈 생성과 `"is a"`가 정답이 되어 버린다.
    """
    pred = normalize_text(prediction)
    gold = normalize_text(target)
    if not pred or not gold:
        return False
    if pred == gold:
        return True
    gold_word = gold.split(" ")[-1].strip(_STRIP)
    return bool(gold_word) and pred.split(" ")[-1].strip(_STRIP) == gold_word


def exact_match(
    predictions: Sequence[str], targets: Sequence[str], numeric: bool = True
) -> float:
    """정확도. 길이가 다르면 던진다 — 조용히 자르면 지표가 왜곡된다."""
    if len(predictions) != len(targets):
        raise ValueError(
            f"예측 {len(predictions)}개와 정답 {len(targets)}개의 수가 다르다."
        )
    if not predictions:
        return 0.0
    fn = match_final_number if numeric else match_free_form
    return sum(fn(p, t) for p, t in zip(predictions, targets)) / len(predictions)


__all__ = (
    "clean_numeric",
    "exact_match",
    "extract_final_number",
    "match_final_number",
    "match_free_form",
    "normalize_text",
    "numbers_equal",
)
