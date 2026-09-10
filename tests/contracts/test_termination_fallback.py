"""I5 — 모든 종료 규칙은 m = M_max에서 반드시 정지한다.

제안서 §4.3: 일부 상태가 진동(궤도)하는 사례가 보고되어 있으므로 M_max
폴백을 필수 명세로 둔다.
"""

from __future__ import annotations

import pytest
import torch

from lsrr.core.errors import TerminationFallbackMissing
from lsrr.core.invariants import assert_has_fallback
from lsrr.core.types import TerminationSignals

pytestmark = pytest.mark.contract


class _NoFallback:
    m_max = None


def test_rule_without_m_max_rejected():
    with pytest.raises(TerminationFallbackMissing, match="M_max 폴백"):
        assert_has_fallback(_NoFallback())


def test_zero_m_max_rejected():
    class Zero:
        m_max = 0

    with pytest.raises(TerminationFallbackMissing):
        assert_has_fallback(Zero())


def test_valid_rule_passes(dummy_cfg):
    from lsrr.builder import build_slots

    assert_has_fallback(build_slots(dummy_cfg, d_in=32, num_layers=6, build_encoder=False).termination)


def test_converging_sequence_stops_early(dummy_cfg):
    from lsrr.builder import build_slots

    rule = build_slots(dummy_cfg, d_in=32, num_layers=6, build_encoder=False).termination
    rule.reset(2, torch.device("cpu"))
    stopped = None
    for m in range(rule.m_max):
        delta = torch.full((2,), 1.0 * (0.1 ** m))
        stopped, _ = rule.should_stop(TerminationSignals(delta_state=delta), m)
        if bool(stopped.all()):
            break
    assert bool(stopped.all()) and m < rule.m_max - 1


def test_oscillating_sequence_falls_back_to_m_max(dummy_cfg):
    """진동하는 상태도 반드시 멈춘다 — 무한 루프 금지."""
    from lsrr.builder import build_slots

    rule = build_slots(dummy_cfg, d_in=32, num_layers=6, build_encoder=False).termination
    rule.reset(2, torch.device("cpu"))
    for m in range(rule.m_max):
        delta = torch.full((2,), 1.0 if m % 2 == 0 else 0.9)  # 절대 ε 아래로 안 감
        stopped, _ = rule.should_stop(TerminationSignals(delta_state=delta), m)
    assert bool(stopped.all()), "M_max에서 정지하지 않았다"
    assert m == rule.m_max - 1


def test_eval_loop_bounded_by_m_max(dummy_model, dummy_batch):
    trace = dummy_model(dummy_batch, is_eval=True)
    assert trace.num_cycles <= dummy_model.termination.m_max
