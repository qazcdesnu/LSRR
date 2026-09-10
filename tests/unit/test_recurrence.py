"""사이클 축 제어 — 스케줄·TBPTT·래칭·러너 (제안서 §4.2, §4.3, §5)."""

from __future__ import annotations

import random

import pytest
import torch
import torch.nn as nn

from lsrr.core.types import TerminationSignals
from lsrr.engine import EngineWrapper
from lsrr.recurrence import (
    CycleRunner,
    DiagnosticsRecorder,
    EarlyExitState,
    FixedSchedule,
    LogNormalSchedule,
    UniformSchedule,
    make_window,
    sample_supervision_cycles,
)
from lsrr.termination import DeltaStateRule, FixedMRule

B, L, D = 4, 6, 8


class _Contracting(nn.Module):
    """지수적으로 수축하는 코어 — 종료 규칙 검증용."""

    def __init__(self, factor: float = 0.1) -> None:
        super().__init__()
        self.factor = factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.factor


class _Oscillating(nn.Module):
    """부호를 뒤집어 진동하는 코어 — M_max 폴백 검증용."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return -x


def _engine(core: nn.Module, alpha: float = 1.0) -> EngineWrapper:
    return EngineWrapper(
        core=core,
        d_model=D,
        damping_alpha=alpha,
        cycle_embedding=False,
        reinject_r0="none",
    )


# ---------------------------------------------------------------- 스케줄


def test_fixed_schedule_is_constant():
    s = FixedSchedule(k=5)
    assert {s.sample_M() for _ in range(20)} == {5}


def test_lognormal_respects_bounds():
    s = LogNormalSchedule(mean=6, std=1.0, max=12, min=2, seed=0)
    values = [s.sample_M() for _ in range(500)]
    assert min(values) >= 2 and max(values) <= 12


def test_lognormal_centres_near_mean():
    s = LogNormalSchedule(mean=6, std=0.4, max=32, seed=0)
    values = sorted(s.sample_M() for _ in range(1000))
    assert 5 <= values[500] <= 7


def test_gamma_weights_emphasise_late_cycles():
    """w_m = γ^(M−m) — '점진 개선 + 후반 완성' 압력 (ADR-006)."""
    w = FixedSchedule(k=4, gamma=0.85).weights(4)
    assert w == sorted(w), "가중이 후반으로 갈수록 커져야 한다"
    assert w[-1] == pytest.approx(0.85)


def test_invalid_gamma_rejected():
    with pytest.raises(ValueError, match="gamma"):
        FixedSchedule(k=4, gamma=1.5)


def test_uniform_schedule_covers_range():
    s = UniformSchedule(min=2, max=5, seed=0)
    assert {s.sample_M() for _ in range(200)} == {2, 3, 4, 5}


# ---------------------------------------------------------------- TBPTT


def test_window_keeps_last_k_cycles():
    w = make_window(M=8, tbptt_k=3)
    assert w.cycles() == [5, 6, 7] and w.k == 3
    assert w.should_detach(4) and not w.should_detach(5)


def test_window_clamps_to_M():
    assert make_window(M=3, tbptt_k=10).cycles() == [0, 1, 2]


def test_supervision_cycles_stay_inside_window():
    """윈도 밖은 detach되어 감독 효과가 없다 (ADR-006)."""
    w = make_window(M=10, tbptt_k=4)
    chosen = sample_supervision_cycles(w, num_cycles=2, rng=random.Random(0))
    assert all(m in w for m in chosen)


def test_supervision_excludes_final_cycle():
    """마지막 사이클은 L_NLL이 이미 감독한다 — 두 번 세면 안 된다."""
    w = make_window(M=6, tbptt_k=4)
    for seed in range(20):
        assert w.M - 1 not in sample_supervision_cycles(w, 2, random.Random(seed))


def test_supervision_empty_when_window_is_one_cycle():
    assert sample_supervision_cycles(make_window(M=1, tbptt_k=1), 2) == []


def test_detach_cuts_gradient_outside_window():
    engine = _engine(nn.Linear(D, D))
    runner = CycleRunner(engine, FixedSchedule(k=4), FixedMRule(m=4, m_max=8), tbptt_k=2)
    R0 = torch.randn(B, L, D, requires_grad=True)
    trace = runner.run_train(R0)
    trace.R_star.sum().backward()
    assert R0.grad is None or float(R0.grad.abs().sum()) == 0.0


# ---------------------------------------------------------------- 래칭


def test_latch_freezes_stopped_samples():
    """정지한 샘플은 더 갱신되지 않는다 — 기록과 최종 상태가 어긋나면 안 된다."""
    R0 = torch.zeros(B, L, D)
    state = EarlyExitState(R0, m_max=5)
    stop = torch.tensor([True, False, False, False])
    state.update(torch.ones(B, L, D), stop, m=0)
    state.update(torch.full((B, L, D), 2.0), torch.ones(B, dtype=torch.bool), m=1)

    assert float(state.R_final[0].mean()) == 1.0  # 0번은 m=0에서 고정
    assert float(state.R_final[1].mean()) == 2.0
    assert state.stopping_cycles.tolist() == [1, 2, 2, 2]


# ---------------------------------------------------------------- 러너


def test_run_train_records_every_cycle():
    runner = CycleRunner(
        _engine(nn.Linear(D, D)), FixedSchedule(k=5), FixedMRule(m=5, m_max=8), tbptt_k=2
    )
    trace = runner.run_train(torch.randn(B, L, D))
    assert trace.num_cycles == 5 and trace.meta["M"] == 5
    assert trace.tbptt_window == (3, 5)


def test_run_train_dispatches_hooks():
    rec = DiagnosticsRecorder()
    runner = CycleRunner(
        _engine(nn.Linear(D, D)), FixedSchedule(k=3), FixedMRule(m=3, m_max=8)
    )
    runner.run_train(torch.randn(B, L, D), hooks=[rec])
    assert len(rec.records) == 3 and len(rec.deltas()) == 3


def test_run_eval_stops_early_on_convergence():
    """수축하는 궤적은 M_max 전에 멈춘다."""
    runner = CycleRunner(
        _engine(_Contracting(0.05)), FixedSchedule(k=8), DeltaStateRule(eps=1e-2, m_max=16)
    )
    trace = runner.run_eval(torch.randn(B, L, D))
    assert trace.num_cycles < 16
    assert float(trace.stopping_cycles.float().mean()) < 16


def test_run_eval_falls_back_at_m_max():
    """진동하는 궤적도 반드시 멈춘다 (I5)."""
    runner = CycleRunner(
        _engine(_Oscillating()), FixedSchedule(k=8), DeltaStateRule(eps=1e-9, m_max=6)
    )
    trace = runner.run_eval(torch.randn(B, L, D))
    assert trace.num_cycles == 6
    assert trace.stopping_cycles.tolist() == [6] * B


def test_delta_trajectory_decreases_for_contracting_core():
    """게이트 ④의 예비 확인 — Δ 궤적의 거시적 감소."""
    runner = CycleRunner(
        _engine(_Contracting(0.3)), FixedSchedule(k=6), FixedMRule(m=6, m_max=8)
    )
    deltas = [d.delta_state for d in runner.run_train(torch.randn(B, L, D)).per_cycle]
    assert deltas[-1] < deltas[0]


def test_runner_rejects_rule_without_fallback():
    class NoFallback:
        m_max = 0

    with pytest.raises(Exception):
        CycleRunner(_engine(nn.Linear(D, D)), FixedSchedule(k=2), NoFallback())


def test_damped_update_matches_proposal_formula():
    """R^(m+1) = (1-α)R^(m) + α·S(R^(m)) — §4.2."""
    alpha = 0.25
    engine = _engine(_Contracting(0.0), alpha=alpha)  # S(x) = 0
    R = torch.randn(B, L, D)
    out = engine.forward_step(R, R, 0)
    assert torch.allclose(out, (1 - alpha) * R, atol=1e-5)


def test_alpha_one_means_no_damping():
    engine = _engine(_Contracting(0.0), alpha=1.0)
    R = torch.randn(B, L, D)
    assert torch.allclose(engine.forward_step(R, R, 0), torch.zeros_like(R), atol=1e-5)


def test_invalid_alpha_rejected():
    with pytest.raises(ValueError, match="damping_alpha"):
        _engine(nn.Linear(D, D), alpha=0.0)


def test_cycle_embedding_conditions_on_m():
    """사이클 임베딩이 실제로 m을 구분한다 (재귀 적응 장치, §4.2)."""
    engine = EngineWrapper(
        core=nn.Identity(), d_model=D, damping_alpha=1.0,
        cycle_embedding=True, reinject_r0="none", max_cycles=4,
    )
    torch.nn.init.normal_(engine.cycle_emb.weight, std=1.0)
    R = torch.randn(B, L, D)
    assert not torch.allclose(engine.forward_step(R, R, 0), engine.forward_step(R, R, 1))
