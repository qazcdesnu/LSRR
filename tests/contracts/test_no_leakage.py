"""I6 — 정답은 문맥 인코딩에 노출되지 않고, 패딩은 감독되지 않는다.

누출 시 전 실험이 무효다.
"""

from __future__ import annotations

import pytest
import torch

from lsrr.core.errors import LeakageError
from lsrr.core.invariants import IGNORE_INDEX, assert_labels_masked, assert_no_answer_leakage
from lsrr.core.types import DataSample
from lsrr.data.prompting import PromptSpec, assert_no_answer_text_leakage

pytestmark = pytest.mark.contract


def test_answer_in_question_template_rejected():
    """템플릿 실수는 1차 방어선에서 잡힌다."""
    bad = PromptSpec(question_template="{question} 정답은 42")
    with pytest.raises(LeakageError, match="정답"):
        assert_no_answer_text_leakage([DataSample(question="3 * 14 =", answer="42")], bad)


def test_clean_prompt_passes():
    assert_no_answer_text_leakage(
        [DataSample(question="3 * 14 =", answer="42")], PromptSpec()
    )


def test_token_level_leakage_detected():
    q = torch.tensor([[5, 6, 7, 8, 9]])
    a = torch.tensor([[7, 8]])  # 질문의 부분열
    with pytest.raises(LeakageError, match="정답 토큰열"):
        assert_no_answer_leakage(q, a)


def test_token_level_clean_passes():
    assert_no_answer_leakage(torch.tensor([[5, 6, 7]]), torch.tensor([[41, 42]]))


def test_labels_mask_padding():
    labels = torch.tensor([[1, 2, IGNORE_INDEX], [3, IGNORE_INDEX, IGNORE_INDEX]])
    assert_labels_masked(labels, pad_token_id=0)


def test_unmasked_padding_rejected():
    with pytest.raises(LeakageError, match="IGNORE_INDEX"):
        assert_labels_masked(torch.tensor([[1, 2, 0], [3, 0, 0]]), pad_token_id=0)


def test_encoder_masks_answer_padding(prompt_encoder):
    """서로 길이가 다른 정답에서 패딩이 IGNORE_INDEX로 마스킹된다."""
    samples = [DataSample(question="1 * 1 =", answer="1"),
               DataSample(question="99 * 99 =", answer="9801")]
    enc = prompt_encoder.encode_answers(samples)
    assert (enc["labels"] == IGNORE_INDEX).any()
    assert_labels_masked(enc["labels"], prompt_encoder.tokenizer.pad_token_id)
    assert enc["target_ids"].shape == enc["labels"].shape
    # target_ids는 실제 pad id로 채워진다 (디코더 입력)
    assert (enc["target_ids"] != IGNORE_INDEX).all()


def test_question_encoding_excludes_answer(prompt_encoder, mult_samples):
    """실제 인코딩 결과에 정답 토큰열이 없다."""
    batch = prompt_encoder.encode_batch(mult_samples)
    assert_no_answer_leakage(batch["input_ids"], batch["labels"])
