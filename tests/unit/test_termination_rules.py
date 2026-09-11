"""종료 규칙과 신호 — Ablation B (제안서 §4.3, §7)."""

from __future__ import annotations

import pytest
import torch

from lsrr.core.errors import ConfigError
from lsrr.core.types import TerminationSignals
from lsrr.termination import (
    DeltaStateRule,
    EntropyOutputRule,
    FixedMRule,
    KLOutputRule,
    output_entropy,
    output_kl,
    relative_delta,
    state_delta,
)

B, L, D = 4, 6, 8
DEV = torch.device("cpu")


def _signals(delta=None, kl=None, entropy=None, batch=B):
    return TerminationSignals(
        delta_state=torch.full((batch,), 1.0) if delta is None else delta,
        kl_div=kl,
        entropy=entropy,
    )


# ---------------------------------------------------------------- 신호


def test_state_delta_is_per_sample():
    """배치 평균으로 정지를 결정하면 '난이도별 적응 계산' 주장이 무너진다."""
    R = torch.zeros(B, L, D)
    R_next = torch.zeros(B, L, D)
    R_next[0] = 1.0
    d = state_delta(R, R_next)
    assert d.shape == (B,)
    assert float(d[0]) > 0 and float(d[1]) == 0.0


def test_state_delta_matches_proposal_formula():
    """v2.1 §4.3: Δ = 1/(L·d_ssm) Σ_l ‖r_l' − r_l‖₁ — 전 원소 평균 절대 변화량."""
    R = torch.zeros(1, 3, 4)
    R_next = torch.zeros(1, 3, 4)
    R_next[0, 0, 0] = 3.0
    # L·d = 12, 절대 변화 총합 3.0 → 0.25
    assert float(state_delta(R, R_next)[0]) == pytest.approx(3.0 / 12)


def test_state_delta_is_invariant_to_width():
    """L1/(L·d) 는 폭에 불변이다 — ε 을 d_ssm 마다 재보정할 필요가 줄어든다.

    v1 의 (1/L)Σ‖·‖₂ 는 같은 원소별 변화량에도 d 가 커지면 √d 로 부풀었다.
    """
    vals = []
    for d in (4, 16, 64):
        R = torch.zeros(1, 3, d)
        R_next = torch.full((1, 3, d), 0.1)
        vals.append(float(state_delta(R, R_next)[0]))
    assert all(v == pytest.approx(0.1) for v in vals), vals


def test_state_delta_differs_from_the_v1_definition():
    """v1 의 ε 을 그대로 쓰면 안 된다 (FINDINGS F-024)."""
    torch.manual_seed(0)
    R = torch.randn(2, 12, 256)
    R_next = R + torch.randn_like(R) * 0.01
    v2 = state_delta(R, R_next)
    v1 = (R_next - R).norm(p=2, dim=-1).mean(dim=-1)   # 구 정의
    assert not torch.allclose(v1, v2, rtol=0.1)
    assert (v1 > v2 * 5).all(), "구 정의가 훨씬 큰 값을 낸다"


def test_relative_delta_normalises_by_state_norm():
    R = torch.full((B, L, D), 10.0)
    R_next = R + 1.0
    assert float(relative_delta(R, R_next).mean()) < float(state_delta(R, R_next).mean())


def test_output_kl_is_zero_for_identical_logits():
    logits = torch.randn(B, 3, 20)
    assert float(output_kl(logits, logits).abs().max()) == pytest.approx(0.0, abs=1e-5)


def test_output_kl_positive_for_different_logits():
    assert float(output_kl(torch.randn(B, 3, 20), torch.randn(B, 3, 20)).min()) > 0


def test_output_kl_none_without_previous():
    assert output_kl(None, torch.randn(B, 3, 20)) is None


def test_entropy_low_for_peaked_distribution():
    peaked = torch.zeros(B, 1, 20)
    peaked[:, :, 0] = 50.0
    flat = torch.zeros(B, 1, 20)
    assert float(output_entropy(peaked).mean()) < float(output_entropy(flat).mean())


# ---------------------------------------------------------------- 폴백 (I5)


@pytest.mark.parametrize(
    "rule",
    [
        FixedMRule(m=5, m_max=5),
        DeltaStateRule(eps=1e-12, m_max=5),
        KLOutputRule(eps=1e-12, m_max=5),
        EntropyOutputRule(eps=-1.0, m_max=5),
    ],
)
def test_every_rule_stops_at_m_max(rule):
    """기반 클래스가 폴백을 합성하므로 어떤 규칙도 우회할 수 없다."""
    rule.reset(B, DEV)
    stopped = None
    for m in range(rule.m_max):
        stopped, _ = rule.should_stop(_signals(), m)
    assert bool(stopped.all())


def test_m_max_below_one_rejected():
    with pytest.raises(ValueError, match="m_max"):
        DeltaStateRule(m_max=0)


# ---------------------------------------------------------------- 규칙


def test_fixed_m_stops_exactly_at_m():
    rule = FixedMRule(m=3, m_max=10)
    rule.reset(B, DEV)
    for m in range(2):
        stopped, _ = rule.should_stop(_signals(), m)
        assert not bool(stopped.any())
    stopped, _ = rule.should_stop(_signals(), 2)
    assert bool(stopped.all())


def test_fixed_m_beyond_m_max_rejected():
    """폴백이 먼저 걸리면 의도한 깊이로 돌지 않는다."""
    with pytest.raises(ConfigError, match="m_max"):
        FixedMRule(m=20, m_max=8)


def test_delta_state_stops_per_sample():
    rule = DeltaStateRule(eps=0.5, m_max=10)
    rule.reset(B, DEV)
    delta = torch.tensor([0.1, 0.9, 0.2, 0.8])
    stopped, _ = rule.should_stop(_signals(delta=delta), 0)
    assert stopped.tolist() == [True, False, True, False]


def test_stop_mask_is_monotonic():
    """한 번 정지한 샘플은 다시 살아나지 않는다."""
    rule = DeltaStateRule(eps=0.5, m_max=10)
    rule.reset(B, DEV)
    rule.should_stop(_signals(delta=torch.tensor([0.1, 0.9, 0.9, 0.9])), 0)
    stopped, _ = rule.should_stop(_signals(delta=torch.tensor([9.0, 9.0, 9.0, 9.0])), 1)
    assert bool(stopped[0])


def test_kl_rule_waits_for_first_signal():
    """첫 사이클에는 이전 로짓이 없으므로 정지하지 않는다."""
    rule = KLOutputRule(eps=1.0, m_max=10)
    rule.reset(B, DEV)
    stopped, _ = rule.should_stop(_signals(kl=None), 0)
    assert not bool(stopped.any())


def test_kl_rule_stops_when_output_converges():
    rule = KLOutputRule(eps=0.1, m_max=10)
    rule.reset(B, DEV)
    stopped, _ = rule.should_stop(_signals(kl=torch.full((B,), 0.01)), 1)
    assert bool(stopped.all())


def test_entropy_rule_stops_when_confident():
    rule = EntropyOutputRule(eps=1.0, m_max=10)
    rule.reset(B, DEV)
    stopped, _ = rule.should_stop(_signals(entropy=torch.tensor([0.5, 2.0, 0.1, 3.0])), 0)
    assert stopped.tolist() == [True, False, True, False]


@pytest.mark.parametrize(
    "rule,needs",
    [
        (FixedMRule(m=2, m_max=8), False),
        (DeltaStateRule(m_max=8), False),
        (KLOutputRule(m_max=8), True),
        (EntropyOutputRule(m_max=8), True),
    ],
)
def test_needs_logits_declared_correctly(rule, needs):
    """설정 검증이 이 선언을 근거로 사이클별 판독 필요 여부를 확인한다."""
    assert rule.needs_logits() is needs


def test_diagnostics_carry_all_signals():
    rule = DeltaStateRule(m_max=8)
    rule.reset(B, DEV)
    _, diag = rule.should_stop(
        _signals(kl=torch.full((B,), 0.3), entropy=torch.full((B,), 2.0)), 0
    )
    assert diag.m == 0 and diag.kl_div == pytest.approx(0.3)
    assert diag.entropy == pytest.approx(2.0)


# ---------------------------------------------------------------- M_min (v2.1 §4.3)

def test_m_min_blocks_early_stopping():
    """M = min{m ∈ {M_min..M_max} | Δ<ε} — 하한 이전에는 규칙이 만족해도 멈추지 않는다."""
    from lsrr.termination.rules import DeltaStateRule

    rule = DeltaStateRule(eps=1e9, m_max=8, m_min=3)  # eps 가 커서 항상 만족
    rule.reset(2, torch.device("cpu"))
    sig = lambda: TerminationSignals(delta_state=torch.zeros(2))

    for m in range(2):  # m+1 = 1, 2 < m_min=3
        stopped, _ = rule.should_stop(sig(), m)
        assert not stopped.any(), f"m={m} 에서 하한을 무시하고 멈췄다"

    stopped, _ = rule.should_stop(sig(), 2)  # m+1 = 3 == m_min
    assert stopped.all()


def test_m_max_fallback_beats_m_min():
    """하한과 상한이 충돌하면 상한이 이긴다 (I5 는 우회 불가)."""
    from lsrr.termination.rules import DeltaStateRule

    rule = DeltaStateRule(eps=0.0, m_max=3, m_min=3)  # eps=0 이라 규칙은 절대 불만족
    rule.reset(2, torch.device("cpu"))
    for m in range(2):
        stopped, _ = rule.should_stop(
            TerminationSignals(delta_state=torch.ones(2)), m
        )
        assert not stopped.any()
    stopped, _ = rule.should_stop(TerminationSignals(delta_state=torch.ones(2)), 2)
    assert stopped.all(), "m_max 폴백이 걸려야 한다"


def test_m_min_defaults_to_one():
    """기본값은 v1 거동과 같다 — 하한을 쓰지 않는 설정이 바뀌지 않는다."""
    from lsrr.termination.rules import DeltaStateRule

    assert DeltaStateRule(eps=1e-3, m_max=8).m_min == 1


def test_impossible_m_min_is_fatal():
    from lsrr.termination.rules import DeltaStateRule

    with pytest.raises(ValueError, match="m_min"):
        DeltaStateRule(eps=1e-3, m_max=4, m_min=5)
    with pytest.raises(ValueError, match="m_min"):
        DeltaStateRule(eps=1e-3, m_max=4, m_min=0)


def test_m_min_applies_to_every_rule():
    """하한도 기반 클래스가 소유하므로 규칙마다 다시 구현하지 않는다."""
    from lsrr.termination.rules import (
        DeltaStateRule,
        EntropyOutputRule,
        FixedMRule,
        KLOutputRule,
    )

    for cls, kw in (
        (DeltaStateRule, {"eps": 1e9}),
        (KLOutputRule, {"eps": 1e9}),
        (EntropyOutputRule, {"eps": 1e9}),
        (FixedMRule, {"m": 1}),
    ):
        rule = cls(m_max=8, m_min=3, **kw)
        rule.reset(1, torch.device("cpu"))
        sig = TerminationSignals(
            delta_state=torch.zeros(1), kl_div=torch.zeros(1), entropy=torch.zeros(1)
        )
        stopped, _ = rule.should_stop(sig, 0)
        assert not stopped.any(), f"{cls.__name__} 이 하한을 무시했다"
