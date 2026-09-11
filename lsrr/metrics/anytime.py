"""사이클별 정확도 곡선 — Phase 0 게이트 ② (제안서 §5의 부산물).

    "매 사이클 답을 읽을 수 있으므로, M을 늘려 가며 정확도를 재면 **anytime 곡선**이
     나온다. 이 곡선이 오르지 않으면 반복이 기여하지 않는다는 뜻이다."

게이트 ②는 이 곡선의 **단조성**을 본다. 엄격한 단조 증가를 요구하지 않는다 —
사이클 하나가 잡음으로 내려앉는 것과 곡선이 평평한 것은 다른 문제다. 그래서
(i) 처음 대비 마지막의 개선폭과 (ii) 스피어만 순위 상관을 함께 본다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

#: 마지막 사이클이 첫 사이클보다 이만큼은 좋아야 한다 (절대 정확도 차).
DEFAULT_MIN_IMPROVEMENT = 0.02
#: 사이클 인덱스와 정확도의 순위 상관이 이 값 이상이어야 한다.
DEFAULT_MIN_TREND = 0.5


@dataclass(frozen=True)
class AnytimeCurve:
    """사이클별 정확도와 그 추세."""

    cycles: tuple[int, ...]
    accuracy: tuple[float, ...]

    @property
    def improvement(self) -> float:
        """마지막 − 처음."""
        return self.accuracy[-1] - self.accuracy[0]

    @property
    def best(self) -> tuple[int, float]:
        """가장 좋은 사이클과 그 정확도."""
        i = max(range(len(self.accuracy)), key=lambda k: self.accuracy[k])
        return self.cycles[i], self.accuracy[i]

    @property
    def trend(self) -> float:
        """사이클 인덱스와 정확도의 스피어만 순위 상관."""
        return _spearman(list(self.cycles), list(self.accuracy))

    def is_increasing(
        self,
        min_improvement: float = DEFAULT_MIN_IMPROVEMENT,
        min_trend: float = DEFAULT_MIN_TREND,
    ) -> bool:
        """게이트 ②의 판정. 개선폭과 추세를 **둘 다** 요구한다."""
        if len(self.accuracy) < 2:
            return False
        return self.improvement >= min_improvement and self.trend >= min_trend


def _rank(values: Sequence[float]) -> list[float]:
    """동점은 평균 순위."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """스피어만 순위 상관. 표본이 2개 미만이거나 분산이 0이면 0."""
    if len(xs) < 2:
        return 0.0
    rx, ry = _rank(xs), _rank(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def build_curve(per_cycle_accuracy: dict[int, float]) -> AnytimeCurve:
    """`{사이클: 정확도}` → 곡선. 사이클 오름차순으로 정렬한다."""
    if not per_cycle_accuracy:
        raise ValueError("빈 사이클별 정확도로는 곡선을 만들 수 없다.")
    items = sorted(per_cycle_accuracy.items())
    return AnytimeCurve(
        cycles=tuple(m for m, _ in items),
        accuracy=tuple(float(a) for _, a in items),
    )


__all__ = (
    "AnytimeCurve",
    "DEFAULT_MIN_IMPROVEMENT",
    "DEFAULT_MIN_TREND",
    "build_curve",
)
