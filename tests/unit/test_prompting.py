"""프롬프트 규약과 데이터셋."""

from __future__ import annotations

import pytest
import torch

from lsrr.core.invariants import IGNORE_INDEX
from lsrr.core.registry import DATA_REGISTRY
from lsrr.core.types import DataSample
from lsrr.data.schema import normalize_split


def test_split_aliases_normalized():
    assert normalize_split("valid") == "val"
    assert normalize_split("validation") == "val"
    assert normalize_split("test") == "test"


def test_unknown_split_rejected():
    with pytest.raises(ValueError, match="split"):
        normalize_split("holdout")


def test_questions_are_left_padded(prompt_encoder):
    """ADR-011: 마지막 실토큰이 항상 인덱스 -1에 놓여야 한다."""
    samples = [
        DataSample(question="1 * 1 =", answer="1"),
        DataSample(question="123456 * 654321 =", answer="0"),
    ]
    enc = prompt_encoder.encode_questions(samples)
    assert bool(enc["attention_mask"][:, -1].all())
    assert not bool(enc["attention_mask"][:, 0].all())


def test_answers_are_right_padded_with_ignore_labels(prompt_encoder):
    samples = [
        DataSample(question="q", answer="1"),
        DataSample(question="q", answer="123456"),
    ]
    enc = prompt_encoder.encode_answers(samples)
    assert enc["labels"][0, -1] == IGNORE_INDEX
    assert enc["answer_lens"][0] < enc["answer_lens"][1]


def test_target_ids_and_labels_differ_in_padding(prompt_encoder):
    """target_ids는 디코더 입력(실제 pad id), labels는 손실 타깃(IGNORE_INDEX)."""
    samples = [
        DataSample(question="q", answer="1"),
        DataSample(question="q", answer="123456"),
    ]
    enc = prompt_encoder.encode_answers(samples)
    assert (enc["target_ids"] != IGNORE_INDEX).all()
    assert (enc["labels"] == IGNORE_INDEX).any()


def test_eos_appended(prompt_encoder):
    enc = prompt_encoder.encode_answers([DataSample(question="q", answer="42")])
    assert prompt_encoder.tokenizer.eos_token_id in enc["target_ids"][0].tolist()


def test_decode_truncates_at_eos(prompt_encoder):
    eos = prompt_encoder.tokenizer.eos_token_id
    head = prompt_encoder.tokenizer("42")["input_ids"][0]
    ids = torch.tensor([[head, eos, 100, 200]])
    assert prompt_encoder.decode(ids)[0] == prompt_encoder.tokenizer.decode([head])


def test_multiplication_scoring_ignores_formatting():
    ds = DATA_REGISTRY.build({"type": "multiplication", "digits": 2})
    assert ds.score(" 4,664 ", "4664", {})
    assert not ds.score("4665", "4664", {})


def test_multiplication_splits_disjoint():
    """split마다 시드 오프셋이 달라 학습·평가 표본이 겹치지 않는다."""
    ds = DATA_REGISTRY.build({"type": "multiplication", "digits": 4})
    tr = {s.question for s in ds.get_split("train")}
    te = {s.question for s in ds.get_split("test")}
    assert len(tr & te) / len(te) < 0.05


def test_multiplication_is_deterministic():
    a = DATA_REGISTRY.build({"type": "multiplication", "digits": 3, "seed": 7})
    b = DATA_REGISTRY.build({"type": "multiplication", "digits": 3, "seed": 7})
    assert [s.question for s in a.get_split("val")] == [
        s.question for s in b.get_split("val")
    ]
