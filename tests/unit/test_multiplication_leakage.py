"""1자리 곱셈의 정답 누출 (I6).

`1 * 7 = 7` 처럼 **정답이 질문에 그대로 보이는** 조합이 1자리에서 21% 나온다.
그대로 학습하면 복사만으로 21%를 맞혀 정확도가 무의미해지고, Phase 0 게이트가
그 위에서 판정하게 된다. `min_operand`가 그 조합을 배제한다.
"""

from __future__ import annotations

import pytest

from lsrr.core.errors import ConfigError, LeakageError
from lsrr.data.datasets.multiplication import MultiplicationDataset

SIZES = {"train": 500, "val": 100, "test": 100}


def _leaks(samples) -> list:
    return [s for s in samples if s.answer in s.question]


def test_single_digit_leaks_without_a_floor():
    """진단 과제를 고르기 전에 알아야 할 사실 — 계약이 잡기 전에 데이터가 이미 샌다."""
    ds = MultiplicationDataset(digits=1, sizes=SIZES)
    leaked = _leaks(ds.get_split("train"))
    assert leaked, "1자리는 누출 조합이 존재한다"
    # 1 * n 과 n * 1 계열이다.
    assert all(s.meta["a"] == 1 or s.meta["b"] == 1 for s in leaked)


def test_min_operand_removes_every_leak():
    ds = MultiplicationDataset(digits=1, sizes=SIZES, min_operand=2)
    for split in ("train", "val", "test"):
        assert _leaks(ds.get_split(split)) == []


def test_min_operand_keeps_the_task_intact():
    """64조합이 남고, 학습 표본이 그것을 모두 덮는다."""
    ds = MultiplicationDataset(digits=1, sizes={"train": 2000, "val": 100, "test": 100},
                               min_operand=2)
    pairs = {(s.meta["a"], s.meta["b"]) for s in ds.get_split("train")}
    assert len(pairs) == 64
    assert all(2 <= a <= 9 and 2 <= b <= 9 for a, b in pairs)


def test_two_digit_needs_no_floor():
    """2자리는 8,100조합 전부 깨끗하다 — 하한이 불필요하다."""
    ds = MultiplicationDataset(digits=2, sizes=SIZES)
    assert _leaks(ds.get_split("train")) == []


def test_impossible_floor_is_fatal():
    with pytest.raises(ConfigError, match="상한"):
        MultiplicationDataset(digits=1, sizes=SIZES, min_operand=20).get_split("train")


def test_splits_stay_disjoint_under_the_floor():
    """하한을 걸어도 split 시드 오프셋은 유지된다."""
    ds = MultiplicationDataset(digits=2, sizes=SIZES, min_operand=20)
    tr = {(s.meta["a"], s.meta["b"]) for s in ds.get_split("train")}
    te = {(s.meta["a"], s.meta["b"]) for s in ds.get_split("test")}
    # 완전 분리는 표본 추출이라 보장되지 않지만, 같은 시퀀스여서는 안 된다.
    assert tr != te


def test_prompt_contract_rejects_leaking_data():
    """I6 는 데이터가 새면 로드 시점에 거부한다 — 학습을 낭비시키지 않는다."""
    from lsrr.core.types import DataSample
    from lsrr.data.prompting import PromptSpec, assert_no_answer_text_leakage

    spec = PromptSpec(max_question_tokens=32, max_answer_tokens=12)
    leaking = [DataSample(question="1 * 7 =", answer="7", meta={})]
    clean = [DataSample(question="3 * 7 =", answer="21", meta={})]

    with pytest.raises(LeakageError, match="정답"):
        assert_no_answer_text_leakage(leaking, spec)
    assert_no_answer_text_leakage(clean, spec)


def test_min_operand_data_passes_the_prompt_contract():
    """하한을 건 1자리 데이터가 실제로 계약을 통과한다."""
    from lsrr.data.prompting import PromptSpec, assert_no_answer_text_leakage

    spec = PromptSpec(max_question_tokens=32, max_answer_tokens=12)
    ds = MultiplicationDataset(digits=1, sizes=SIZES, min_operand=2)
    assert_no_answer_text_leakage(ds.get_split("train"), spec)

    with pytest.raises(LeakageError):
        assert_no_answer_text_leakage(
            MultiplicationDataset(digits=1, sizes=SIZES).get_split("train"), spec
        )
