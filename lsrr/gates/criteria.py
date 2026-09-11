"""게이트 술어 (ADR-008, 제안서 §6.3).

각 술어는 **판정 데이터를 받아 PASS/FAIL과 그 근거를 낸다**. 데이터를 만들지
않는다 — 실험을 돌리는 것과 판정하는 것을 분리해야 "실패한 실험 앞에서 기준을
조금 낮추는" 일이 구조적으로 불가능해진다.

임계값은 전부 인자로 노출한다. 다만 **기본값이 정본**이며, 실험 전에 설정으로
고정해 런 스냅샷에 포함한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from lsrr.analysis.collapse import CollapseReport
from lsrr.metrics.aggregate import welch_ttest
from lsrr.metrics.anytime import AnytimeCurve
from lsrr.termination.behavior import BehaviorReport, behavior_counts


@dataclass(frozen=True)
class GateResult:
    """게이트 하나의 판정."""

    gate_id: str
    name: str
    passed: bool
    detail: str
    is_kill_switch: bool = False
    evaluated: bool = True
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        """판정 불가는 FAIL 과 구분한다 — 킬 스위치는 **측정된** 실패에서만 발동한다."""
        if not self.evaluated:
            return "N/A"
        return "PASS" if self.passed else "FAIL"


def gate_no_collapse(
    report: CollapseReport,
    min_rank: float = 2.0,
    max_similarity: float = 0.95,
    min_variance_ratio: float = 0.1,
) -> GateResult:
    """① 자명해 붕괴 없음.

    붕괴한 상태에서는 M을 늘려도 정확도가 오를 리 없으므로, ②·③보다 먼저
    판정해야 하는 위생 검사다.
    """
    collapsed = report.is_collapsed(min_rank, max_similarity, min_variance_ratio)
    reasons = report.reasons(min_rank, max_similarity, min_variance_ratio)
    detail = (
        f"붕괴 징후: {', '.join(reasons)}"
        if collapsed
        else (
            f"유효 랭크 {report.effective_rank:.2f}, "
            f"평균 유사도 {report.mean_similarity:.3f}"
            + (
                f", 층간 분산비 {report.variance_ratio:.3f}"
                if report.variance_ratio is not None
                else ""
            )
        )
    )
    return GateResult(
        gate_id="①",
        name="자명해 붕괴 없음",
        passed=not collapsed,
        detail=detail,
        evidence={
            "effective_rank": report.effective_rank,
            "mean_similarity": report.mean_similarity,
            "variance_ratio": report.variance_ratio,
        },
    )


def gate_anytime_increasing(
    curve: AnytimeCurve,
    min_improvement: float = 0.02,
    min_trend: float = 0.5,
) -> GateResult:
    """② M 증가 → 정확도 증가.

    실패하면 ADR-006의 재검토 조건이 발동한다 — 과잉 감독으로 반복이 형해화한
    경우를 먼저 의심하고 λ_ds·γ를 낮춘다.
    """
    ok = curve.is_increasing(min_improvement, min_trend)
    best_m, best_acc = curve.best
    return GateResult(
        gate_id="②",
        name="M 증가 → 정확도 증가",
        passed=ok,
        detail=(
            f"개선폭 {curve.improvement:+.4f} (기준 ≥{min_improvement}), "
            f"추세 {curve.trend:+.2f} (기준 ≥{min_trend}), "
            f"최고 m={best_m}에서 {best_acc:.4f}"
        ),
        evidence={
            "cycles": list(curve.cycles),
            "accuracy": list(curve.accuracy),
            "improvement": curve.improvement,
            "trend": curve.trend,
        },
    )


def gate_beats_onepass(
    hydra_seeds: Sequence[float],
    mlp_seeds: Sequence[float],
    alpha: float = 0.05,
    min_effect: float = 0.5,
    min_difference: float = 0.01,
) -> GateResult:
    """③ H+MLP 1회 통과 대비 유의한 우위 — **킬 스위치**.

    시드가 조건당 2개 미만이면 판정하지 않고 FAIL로 둔다. 단일 시드 판정을
    허용하면 잡음으로 킬 스위치가 통과할 수 있다.
    """
    try:
        cmp = welch_ttest(hydra_seeds, mlp_seeds)
    except ValueError as e:
        return GateResult(
            gate_id="③",
            name="H+MLP 1회 통과 대비 유의한 우위",
            passed=False,
            detail=f"판정 불가: {e}",
            is_kill_switch=True,
            evaluated=False,
            evidence={"hydra_n": len(hydra_seeds), "mlp_n": len(mlp_seeds)},
        )

    ok = cmp.is_significant(
        alpha=alpha, min_effect=min_effect, min_difference=min_difference
    )
    return GateResult(
        gate_id="③",
        name="H+MLP 1회 통과 대비 유의한 우위",
        passed=ok,
        detail=(
            f"hydra_qs {cmp.treatment.mean:.4f}±{cmp.treatment.std:.4f} "
            f"(n={cmp.treatment.n}) vs mlp_onepass {cmp.control.mean:.4f}±"
            f"{cmp.control.std:.4f} (n={cmp.control.n}) → "
            f"차이 {cmp.difference:+.4f} (기준 ≥{min_difference}), "
            f"p={cmp.p_value:.4f} (기준 <{alpha}), "
            f"효과크기 d={cmp.effect_size:.2f} (기준 ≥{min_effect})"
        ),
        is_kill_switch=True,
        evidence={
            "difference": cmp.difference,
            "p_value": cmp.p_value,
            "effect_size": cmp.effect_size,
            "hydra_mean": cmp.treatment.mean,
            "mlp_mean": cmp.control.mean,
        },
    )


def gate_delta_decreasing(
    reports: Sequence[BehaviorReport],
    min_converged_ratio: float = 0.6,
) -> GateResult:
    """④ Δ⁽ᵐ⁾ 궤적의 거시적 감소.

    `converged` 비율로 판정한다. `oscillating`이 많으면 `stability` 사다리를
    켤 후보이고, `drifting`이 많으면 M_max가 짧았을 수 있다 (ADR-007).
    """
    if not reports:
        return GateResult(
            gate_id="④",
            name="Δ 궤적의 거시적 감소",
            passed=False,
            detail="판정 불가: Δ 궤적이 하나도 없다.",
            evaluated=False,
        )

    counts = behavior_counts(reports)
    ratio = counts["converged"] / len(reports)
    settled = sum(1 for r in reports if r.settled) / len(reports)
    return GateResult(
        gate_id="④",
        name="Δ 궤적의 거시적 감소",
        passed=ratio >= min_converged_ratio,
        detail=(
            f"converged {counts['converged']}/{len(reports)} = {ratio:.2f} "
            f"(기준 ≥{min_converged_ratio}), "
            f"oscillating {counts['oscillating']}, drifting {counts['drifting']} "
            f"| ε 아래 안착 {settled:.2f}"
        ),
        evidence={
            "counts": counts,
            "converged_ratio": ratio,
            "settled_ratio": settled,
        },
    )


__all__ = (
    "GateResult",
    "gate_anytime_increasing",
    "gate_beats_onepass",
    "gate_delta_decreasing",
    "gate_no_collapse",
)
