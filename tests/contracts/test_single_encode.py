"""I2 — 샘플 배치당 백본 인코딩은 정확히 1회.

사이클 루프 안에서 백본에 재진입하면 Coconut형 반복 호출이 되어 본 연구의
효율 주장이 사라진다. 재인코딩 외부 루프는 ADR-010의 명시적 예외 플래그
하에서만 허용된다.
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from lsrr.core.errors import MultipleEncodeError
from lsrr.core.invariants import EncodeCounter, encode_guard

pytestmark = pytest.mark.contract


def test_single_encode_allowed():
    counter = EncodeCounter()
    with encode_guard(counter):
        counter.record()
    assert counter.count == 1


def test_second_encode_raises():
    counter = EncodeCounter()
    counter.record()
    with pytest.raises(MultipleEncodeError, match="1회 호출"):
        counter.record()


def test_reset_between_batches():
    counter = EncodeCounter()
    for _ in range(3):
        counter.reset()
        counter.record()
    assert counter.count == 1 and counter.total == 3


def test_reencoding_allowed_only_under_explicit_flag():
    """ADR-010: 플래그 없이는 열리지 않는다."""
    counter = EncodeCounter(allow_reencoding=True)
    counter.record()
    counter.record()
    assert counter.count == 2


def test_model_forward_encodes_once(dummy_model, dummy_batch):
    dummy_model(dummy_batch)
    assert dummy_model.encode_counter.count == 1
    assert dummy_model.encoder.calls == 1


def test_model_forward_twice_resets_counter(dummy_model, dummy_batch):
    dummy_model(dummy_batch)
    dummy_model(dummy_batch)
    assert dummy_model.encode_counter.count == 1
    assert dummy_model.encode_counter.total == 2


def test_extra_encode_within_step_raises(dummy_model, dummy_batch):
    """스텝 도중 추가 인코딩은 즉시 실패한다."""
    dummy_model.encode_counter.reset()
    dummy_model.encode(dummy_batch["input_ids"])
    with pytest.raises(MultipleEncodeError):
        dummy_model.encode(dummy_batch["input_ids"])
