"""시드 집계와 유의성 검정 — Phase 0 게이트 ③ (ADR-009).

게이트 ③은 킬 스위치다. "hydra가 mlp보다 높았다"를 시드 하나로 판정하면 잡음으로
킬 스위치를 통과하거나 발동시킬 수 있으므로, **다중 시드 + 유의성 검정**을 요구한다.

SciPy에 의존하지 않는다 — 이 저장소의 의존은 `pyproject.toml`에 열거된 것뿐이고,
검정 하나 때문에 늘리지 않는다. Welch t 검정과 정규 근사 p값으로 충분하다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class SeedSummary:
    """한 조건의 시드별 결과 요약."""

    values: tuple[float, ...]

    @property
    def n(self) -> int:
        return len(self.values)

    @property
    def mean(self) -> float:
        return sum(self.values) / self.n if self.n else float("nan")

    @property
    def std(self) -> float:
        """표본 표준편차 (n−1). 시드가 1개면 0."""
        if self.n < 2:
            return 0.0
        m = self.mean
        return math.sqrt(sum((v - m) ** 2 for v in self.values) / (self.n - 1))

    @property
    def stderr(self) -> float:
        return self.std / math.sqrt(self.n) if self.n else float("nan")

    def ci95(self) -> tuple[float, float]:
        """정규 근사 95% 신뢰구간."""
        h = 1.96 * self.stderr
        return (self.mean - h, self.mean + h)


@dataclass(frozen=True)
class ComparisonResult:
    """두 조건의 비교 결과."""

    treatment: SeedSummary
    control: SeedSummary
    t_statistic: float
    p_value: float
    effect_size: float

    @property
    def difference(self) -> float:
        return self.treatment.mean - self.control.mean

    def is_significant(
        self,
        alpha: float = 0.05,
        min_effect: float = 0.0,
        min_difference: float = 0.0,
    ) -> bool:
        """단측 우위 판정.

        네 조건을 **모두** 요구한다: 처치가 더 좋고, p < α이고, 효과크기가 기준
        이상이고, **절대 차이가 기준 이상**.

        절대 차이 하한이 필요한 이유는 효과크기가 비율이기 때문이다 — d는 합동
        표준편차로 나눈 값이라, 시드들이 매우 일관되면 실질적으로 무의미한 차이도
        d가 커진다. 실측: 평균차 0.003(0.3%p)에 시드 3개면 p=0.019, d=1.69 로
        "유의"가 나온다 (F-016). 게이트 ③은 킬 스위치이므로 이 통과를 허용하면
        안 된다.
        """
        return (
            self.difference > 0
            and self.p_value < alpha
            and abs(self.effect_size) >= min_effect
            and self.difference >= min_difference
        )


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def welch_ttest(treatment: Sequence[float], control: Sequence[float]) -> ComparisonResult:
    """Welch t 검정 (등분산 가정 없음), 단측 p값.

    시드 수가 적고 조건별 분산이 다를 수 있으므로 Student가 아니라 Welch다.
    p값은 정규 근사다 — 시드 3~5개에서는 보수적이지 않으므로, 게이트는 p값과
    효과크기를 **함께** 요구한다.
    """
    a, b = SeedSummary(tuple(treatment)), SeedSummary(tuple(control))
    if a.n < 2 or b.n < 2:
        raise ValueError(
            f"유의성 검정에는 조건당 시드가 2개 이상 필요하다 "
            f"(처치 {a.n}개, 대조 {b.n}개). 게이트 ③은 킬 스위치이므로 "
            f"단일 시드 판정을 허용하지 않는다."
        )

    va, vb = a.std**2 / a.n, b.std**2 / b.n
    denom = math.sqrt(va + vb)
    t = (a.mean - b.mean) / denom if denom > 0 else 0.0
    p = 1.0 - _normal_cdf(t) if denom > 0 else 0.5

    pooled = math.sqrt((a.std**2 + b.std**2) / 2)
    d = (a.mean - b.mean) / pooled if pooled > 0 else 0.0

    return ComparisonResult(
        treatment=a, control=b, t_statistic=t, p_value=p, effect_size=d
    )


__all__ = ("ComparisonResult", "SeedSummary", "welch_ttest")
