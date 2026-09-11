"""채점·비용·집계 (제안서 §6, I7).

정확도 프로토콜은 외부 공표치와의 비교가 걸린 부분이라 **느슨해지면 안 된다**.
비용 회계는 백본 1회를 빼면 §6의 효율 주장이 검증 불가능해진다.
"""

from __future__ import annotations

import pytest

from lsrr.core.errors import CostAccountingError
from lsrr.metrics.accuracy import exact_match, match_final_number, match_free_form
from lsrr.metrics.aggregate import SeedSummary, welch_ttest
from lsrr.metrics.anytime import build_curve
from lsrr.metrics.cost import (
    CostReport,
    attention_engine_flops,
    ssm_engine_flops,
    transformer_forward_flops,
)


# ---------------------------------------------------------------- 정확도

@pytest.mark.parametrize(
    "pred,gold,expected",
    [
        ("the answer is 42", "42", True),
        ("#### 42", "42", True),
        ("42.0", "42", True),
        ("1,000", "1000", True),
        ("5 12 7 42", "12", False),   # 중간 등장은 정답이 아니다
        (" 42 999", "42", False),     # 마지막 숫자만 본다
        ("", "42", False),
        ("no digits", "42", False),
    ],
)
def test_numeric_scoring_is_final_answer_only(pred, gold, expected):
    assert match_final_number(pred, gold) is expected


@pytest.mark.parametrize(
    "pred,expected",
    [
        ("Sally is a sterpus.", True),
        ("sterpus", True),
        ("", False),
        ("sally", False),
        ("is a", False),
        ("Sally is a bompus.", False),
    ],
)
def test_free_form_rejects_substrings(pred, expected):
    assert match_free_form(pred, "Sally is a sterpus.") is expected


def test_exact_match_length_mismatch_is_fatal():
    with pytest.raises(ValueError, match="수가 다르다"):
        exact_match(["1", "2"], ["1"])


# ---------------------------------------------------------------- anytime

def test_curve_orders_by_cycle():
    c = build_curve({3: 0.4, 0: 0.1, 1: 0.2})
    assert c.cycles == (0, 1, 3)
    assert c.improvement == pytest.approx(0.3)


def test_curve_reports_best_cycle():
    assert build_curve({0: 0.1, 1: 0.4, 2: 0.2}).best == (1, 0.4)


def test_empty_curve_is_fatal():
    with pytest.raises(ValueError):
        build_curve({})


def test_trend_is_rank_based_not_value_based():
    """순위 상관이므로 값의 스케일에 흔들리지 않는다."""
    a = build_curve({0: 0.1, 1: 0.2, 2: 0.3})
    b = build_curve({0: 0.1, 1: 0.11, 2: 0.9})
    assert a.trend == pytest.approx(b.trend)


# ---------------------------------------------------------------- 집계

def test_seed_summary_statistics():
    s = SeedSummary((0.2, 0.3, 0.4))
    assert s.mean == pytest.approx(0.3)
    assert s.std == pytest.approx(0.1)
    lo, hi = s.ci95()
    assert lo < s.mean < hi


def test_welch_needs_two_seeds_per_condition():
    with pytest.raises(ValueError, match="시드가 2개 이상"):
        welch_ttest([0.3], [0.2, 0.25])


def test_welch_detects_a_real_difference():
    r = welch_ttest([0.31, 0.29, 0.33], [0.21, 0.22, 0.20])
    assert r.difference > 0 and r.p_value < 0.05 and r.is_significant()


def test_welch_is_one_sided():
    """대조가 더 좋으면 유의하지 않다 — 우위 주장만 판정한다."""
    r = welch_ttest([0.21, 0.22, 0.20], [0.31, 0.29, 0.33])
    assert not r.is_significant()
    assert r.p_value > 0.5


def test_effect_size_gate_rejects_tiny_but_significant_differences():
    """시드를 늘리면 아주 작은 차이도 p<0.05 가 된다 — 효과크기가 그것을 막는다."""
    r = welch_ttest([0.2001] * 5 + [0.2002] * 5, [0.2000] * 5 + [0.2001] * 5)
    assert not r.is_significant(min_effect=0.5) or r.effect_size >= 0.5


# ---------------------------------------------------------------- 비용 (I7)

def test_total_flops_requires_the_backbone():
    """백본을 못 쟀는데 총계를 내면 I7 위반이다."""
    report = CostReport(
        backbone_flops=None, memory_flops=1e6,
        engine_flops_per_cycle=1e7, cycles=6.0, readout_flops=1e6,
    )
    assert not report.includes_backbone
    with pytest.raises(CostAccountingError, match="0으로 두지 않는다"):
        _ = report.total_flops
    assert report.as_dict()["total_flops"] is None


def test_total_flops_sums_every_term():
    report = CostReport(
        backbone_flops=1e10, memory_flops=1e6,
        engine_flops_per_cycle=1e7, cycles=6.0, readout_flops=1e6,
    )
    assert report.total_flops == pytest.approx(1e10 + 1e6 + 6e7 + 1e6)
    assert report.as_dict()["includes_backbone"] is True


def test_backbone_dominates_a_refinement_cycle():
    """§6 의 효율 주장이 걸린 비교 — 백본을 빼면 비용의 대부분이 사라진다."""
    backbone = transformer_forward_flops(num_layers=12, d_model=768, seq_len=128)
    cycle = ssm_engine_flops(d_model=768, num_layers_axis=12, n_blocks=2)
    assert backbone > 10 * cycle


def test_attention_is_quadratic_in_the_layer_axis():
    """어텐션의 이차 항이 §4.2 에서 SSM 을 고르는 논거다."""
    small = attention_engine_flops(768, num_layers_axis=12, n_blocks=1, d_ffn=3072)
    large = attention_engine_flops(768, num_layers_axis=48, n_blocks=1, d_ffn=3072)
    ssm_small = ssm_engine_flops(768, num_layers_axis=12, n_blocks=1)
    ssm_large = ssm_engine_flops(768, num_layers_axis=48, n_blocks=1)
    assert (large / small) > (ssm_large / ssm_small)


def test_latency_total_is_none_until_both_halves_are_measured():
    r = CostReport(
        backbone_flops=1e10, memory_flops=0, engine_flops_per_cycle=0,
        cycles=1, readout_flops=0, backbone_latency_ms=300.0,
    )
    assert r.total_latency_ms is None
