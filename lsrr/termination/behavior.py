"""수렴 거동 분류 — Phase 0 게이트 ④와 층화 분석의 축 (제안서 §7).

루프 모델에서 일부 상태가 수렴 대신 **진동(궤도)**하는 사례가 보고되어 있다.
Δ⁽ᵐ⁾ 궤적을 세 거동으로 나눠, 종료 시점 민감도를 거동별로 층화한다.

| 라벨 | 기준 |
|---|---|
| `converged` | Δ가 거시적으로 감소한다 (궤적의 **모양**) |
| `oscillating` | Δ가 주기적으로 반등 — 인접 차분의 부호가 자주 뒤집힌다 |
| `drifting` | Δ가 감소하지 않는다 |

**라벨은 모양을, `settled`는 임계값을 말한다.** ε 아래로 내려갔는지는 궤적의
성질이 아니라 임계값 선택의 문제이고, M_max가 짧으면 잘 수렴하는 궤적도 ε 위에서
끝난다. 둘을 한 라벨에 섞으면 ε을 바꿀 때마다 거동 분류가 통째로 뒤집힌다 —
ε 스윕(`calibration.py`, M6)이 성립하지 않는다.

**"거시적" 감소를 쓰는 이유:** 엄격한 단조 감소를 요구하면 수치 잡음 하나로
`converged`가 무너진다. 실제 궤적은 대체로 감소하다 끝에서 미세하게 반등한다
(M4 관측: 2.85 → … → 0.0086 → 0.0089 → 0.0107). 그래서 전반부·후반부 평균의
비율로 판정하고, 반등의 **빈도**는 별도로 `oscillating`이 잡는다.

이 라벨은 `analysis`와 `reporting`에서 층화 축으로 쓰인다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

Behavior = Literal["converged", "oscillating", "drifting"]

#: 후반 평균이 전반 평균의 이 비율 아래로 내려가면 "거시적으로 감소했다".
DEFAULT_DECAY_RATIO = 0.5
#: 인접 차분의 부호 반전 비율이 이 값을 넘으면 진동으로 본다.
DEFAULT_REVERSAL_RATE = 0.34


@dataclass(frozen=True)
class BehaviorReport:
    """궤적 하나의 분류 결과와 그 근거."""

    label: Behavior
    decay_ratio: float
    reversal_rate: float
    final_delta: float
    num_cycles: int
    settled: bool

    @property
    def macroscopically_decreasing(self) -> bool:
        """게이트 ④의 판정 대상. 안착 여부와 무관한 **모양**의 성질이다."""
        return self.label == "converged"


def _reversal_rate(deltas: Sequence[float]) -> float:
    """인접 차분의 부호가 뒤집히는 비율. 진동의 지표."""
    diffs = [b - a for a, b in zip(deltas, deltas[1:])]
    if len(diffs) < 2:
        return 0.0
    reversals = sum(
        1 for a, b in zip(diffs, diffs[1:]) if (a > 0) != (b > 0) and a != 0 and b != 0
    )
    return reversals / (len(diffs) - 1)


def _decay_ratio(deltas: Sequence[float]) -> float:
    """후반부 평균 / 전반부 평균. 작을수록 잘 감소했다."""
    half = max(1, len(deltas) // 2)
    first = sum(deltas[:half]) / half
    second = sum(deltas[half:]) / max(1, len(deltas) - half)
    if first == 0.0:
        return 0.0 if second == 0.0 else float("inf")
    return second / first


def classify_trajectory(
    deltas: Sequence[float],
    eps: float,
    decay_ratio_threshold: float = DEFAULT_DECAY_RATIO,
    reversal_threshold: float = DEFAULT_REVERSAL_RATE,
) -> BehaviorReport:
    """Δ⁽ᵐ⁾ 궤적 하나를 분류한다.

    Args:
        deltas: 사이클 순서의 Δ 값들.
        eps: 종료 임계값. `converged` 판정의 안착 기준.

    Raises:
        ValueError: 궤적이 비어 있으면. 조용히 기본 라벨을 주면 게이트 ④가
            측정 없이 통과할 수 있다 (규약 §3).
    """
    if not deltas:
        raise ValueError("빈 Δ 궤적은 분류할 수 없다.")

    values = [float(d) for d in deltas]
    decay = _decay_ratio(values)
    reversal = _reversal_rate(values)
    final = values[-1]

    settled = final < eps

    if len(values) == 1:
        # 사이클 1회로는 모양을 말할 수 없다. 안착했으면 수렴으로, 아니면 보류.
        label: Behavior = "converged" if settled else "drifting"
    elif reversal > reversal_threshold:
        label = "oscillating"
    elif decay < decay_ratio_threshold:
        # 거시적으로 감소한다. ε 위에서 끝났더라도 모양은 수렴형이다 —
        # M_max 가 짧았을 뿐일 수 있고, 그 판단은 `settled` 가 따로 말한다.
        label = "converged"
    else:
        label = "drifting"

    return BehaviorReport(
        label=label,
        decay_ratio=decay,
        reversal_rate=reversal,
        final_delta=final,
        num_cycles=len(values),
        settled=settled,
    )


def classify_batch(
    trajectories: Sequence[Sequence[float]], eps: float
) -> list[BehaviorReport]:
    """여러 궤적을 분류한다."""
    return [classify_trajectory(t, eps) for t in trajectories]


def behavior_counts(reports: Sequence[BehaviorReport]) -> dict[str, int]:
    """라벨별 개수. 층화 분석과 게이트 보고에 쓴다."""
    counts = {"converged": 0, "oscillating": 0, "drifting": 0}
    for r in reports:
        counts[r.label] += 1
    return counts


__all__ = (
    "Behavior",
    "BehaviorReport",
    "DEFAULT_DECAY_RATIO",
    "DEFAULT_REVERSAL_RATE",
    "classify_trajectory",
    "classify_batch",
    "behavior_counts",
)
