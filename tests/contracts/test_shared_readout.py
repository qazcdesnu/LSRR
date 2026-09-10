"""I3 — 모든 사이클이 동일한 판독 경로 인스턴스를 쓴다.

제안서 §5: 풀링·융합·주입의 판독 경로는 전 사이클 공유(사이클별 헤드 금지)
— 임의 시점 종료에도 답이 판독되는 anytime 성질의 원천이다.
"""

from __future__ import annotations

import pytest

from lsrr.core.errors import SharedReadoutViolation
from lsrr.core.invariants import assert_shared_readout

pytestmark = pytest.mark.contract


def test_same_instance_passes():
    obj = object()
    assert_shared_readout([obj, obj, obj])


def test_distinct_instances_raise():
    with pytest.raises(SharedReadoutViolation, match="사이클별 헤드 금지"):
        assert_shared_readout([object(), object()])


def test_none_entries_ignored():
    obj = object()
    assert_shared_readout([obj, None, obj])


def test_model_readout_is_single_instance(dummy_model, dummy_batch):
    """모델이 중간 사이클과 최종 판독에 같은 객체를 쓴다."""
    from lsrr.core.types import ContextBundle

    context = dummy_model.encode(dummy_batch["input_ids"])
    R0 = dummy_model.build_memory(context)

    used = []
    for m in range(3):
        dummy_model.read(R0, context, m=m)
        used.append(dummy_model.readout)

    assert_shared_readout(used)
    assert dummy_model.readout.call_count == 3


def test_readout_path_identical_for_intermediate_and_final(dummy_model, dummy_batch):
    """중간 상태와 최종 상태의 판독 경로가 같은 타입·같은 결과 구조를 낸다."""
    context = dummy_model.encode(dummy_batch["input_ids"])
    R0 = dummy_model.build_memory(context)
    trace = dummy_model.refine(R0)

    mid = dummy_model.read(R0, context, m=0)
    final = dummy_model.read(trace.R_star, context)
    assert type(mid) is type(final)
    assert mid.h_fusion.shape == final.h_fusion.shape
