"""깊은 감독 (ADR-006).

레거시 대비 두 가지가 달라졌고, 이 파일이 그 둘을 고정한다.

- **윈도 밖 감독 금지** — detach 된 상태를 감독하면 엔진에 그래디언트가 가지
  않는다. 계산만 낭비하고 손실 스케일만 흔든다.
- **γ 가중** — 후반 사이클에 더 큰 압력을 준다 (§5의 '점진 개선 + 후반 완성').
"""

from __future__ import annotations

import pytest
import torch

from lsrr.core.invariants import IGNORE_INDEX
from lsrr.core.types import ReadoutResult, ReasoningTrace
from lsrr.objectives.deep_supervision import DeepSupervision

B, T, V = 2, 4, 16


def _trace(cycles, window, seed=0):
    """지정한 사이클들에 판독 결과가 달린 트레이스."""
    torch.manual_seed(seed)
    return ReasoningTrace(
        R_star=torch.randn(B, 6, 8),
        tbptt_window=window,
        per_cycle_readout=[
            ReadoutResult(logits=torch.randn(B, T, V), m=m) for m in cycles
        ],
    )


def _batch():
    labels = torch.randint(0, V, (B, T))
    labels[:, -1] = IGNORE_INDEX
    return {"labels": labels}


def test_supervises_only_cycles_inside_the_window():
    """윈도 밖(0,1)은 detach 되어 있으므로 감독 대상이 아니다."""
    trace = _trace(cycles=[0, 1, 2, 3, 4], window=(2, 5))
    out = DeepSupervision(num_cycles=10)(trace, _batch())
    # 윈도 [2,5) 에서 마지막(4)은 L_NLL 이 감독하므로 2, 3 만 남는다.
    assert int(out["deep_sup_cycles"]) == 2


def test_final_cycle_excluded_by_default():
    """마지막 사이클을 넣으면 L_NLL 과 같은 항을 두 번 세게 된다."""
    trace = _trace(cycles=[2, 3, 4], window=(2, 5))
    assert int(DeepSupervision(num_cycles=10)(trace, _batch())["deep_sup_cycles"]) == 2
    assert (
        int(
            DeepSupervision(num_cycles=10, include_final=True)(trace, _batch())[
                "deep_sup_cycles"
            ]
        )
        == 3
    )


def test_gamma_weights_later_cycles_more():
    """w_m = γ^(M−m) 이므로 후반 사이클의 기여가 크다."""
    torch.manual_seed(0)
    batch = _batch()
    early = ReadoutResult(logits=torch.randn(B, T, V), m=2)
    late = ReadoutResult(logits=torch.randn(B, T, V), m=3)

    def loss_with(results):
        tr = ReasoningTrace(
            R_star=torch.randn(B, 6, 8), tbptt_window=(2, 5), per_cycle_readout=results
        )
        return float(DeepSupervision(gamma=0.5, num_cycles=10)(tr, batch)["loss"])

    # 같은 두 사이클을 서로 바꿔 달면 가중이 달라지므로 손실이 달라져야 한다.
    a = loss_with([ReadoutResult(logits=early.logits, m=2), ReadoutResult(logits=late.logits, m=3)])
    b = loss_with([ReadoutResult(logits=late.logits, m=2), ReadoutResult(logits=early.logits, m=3)])
    assert a != pytest.approx(b)


def test_gamma_one_is_uniform_weighting():
    """γ=1 이면 레거시와 같은 균등 가중이다."""
    trace = _trace(cycles=[2, 3], window=(2, 5))
    batch = _batch()
    uniform = DeepSupervision(gamma=1.0, num_cycles=10)(trace, batch)["loss"]

    from lsrr.objectives.targets import loss_targets, token_nll

    manual = sum(token_nll(r.logits, loss_targets(batch)) for r in trace.per_cycle_readout) / 2
    assert float(uniform) == pytest.approx(float(manual), abs=1e-6)


def test_num_cycles_caps_the_supervision_set():
    trace = _trace(cycles=[1, 2, 3, 4, 5], window=(1, 6))
    out = DeepSupervision(num_cycles=2)(trace, _batch())
    assert int(out["deep_sup_cycles"]) == 2


def test_no_candidates_yields_zero_not_a_crash():
    """M=1 이거나 윈도에 마지막 사이클만 있으면 감독할 것이 없다."""
    trace = _trace(cycles=[0], window=(0, 1))
    out = DeepSupervision()(trace, _batch())
    assert float(out["loss"]) == 0.0
    assert int(out["deep_sup_cycles"]) == 0


def test_zero_loss_is_reported_not_hidden():
    """조용히 건너뛰지 않고 지표로 드러낸다 (규약 §3)."""
    trace = _trace(cycles=[0], window=(0, 1))
    assert "deep_sup_cycles" in DeepSupervision()(trace, _batch())


def test_gamma_must_be_in_range():
    with pytest.raises(ValueError, match="gamma"):
        DeepSupervision(gamma=0.0)
    with pytest.raises(ValueError, match="gamma"):
        DeepSupervision(gamma=1.5)


def test_gradient_flows_to_the_supervised_logits():
    trace = _trace(cycles=[2, 3, 4], window=(2, 5))
    for r in trace.per_cycle_readout:
        r.logits.requires_grad_(True)
    DeepSupervision(num_cycles=10)(trace, _batch())["loss"].backward()
    # 윈도 안이고 마지막이 아닌 사이클에만 그래디언트가 간다.
    grads = [r.logits.grad is not None for r in trace.per_cycle_readout]
    assert grads == [True, True, False]
