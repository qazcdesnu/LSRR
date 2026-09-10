"""사이클 스케줄 — M 샘플링과 깊은 감독 가중 (제안서 §5, §4.3).

    "학습 시 M을 로그정규 분포에서 샘플링하고 마지막 k 사이클만 역전파한다."

M 무작위 샘플링은 수렴 압력 (i)이다 — 임의 깊이에서 잘려도 정답을 유지해야
하므로 정상 상태가 유리해진다.
"""

from __future__ import annotations

import math
import random
from typing import Any, Optional

from lsrr.core.interfaces import BaseCycleSchedule
from lsrr.core.registry import SCHEDULE_REGISTRY

# `max`/`min`이 생성자 인자 이름으로 가려지므로 내장 함수를 미리 잡아 둔다.
_max, _min = max, min


class _BaseSchedule(BaseCycleSchedule):
    """γ 가중을 공통 제공한다."""

    def __init__(self, gamma: float = 0.85, seed: Optional[int] = None) -> None:
        if not 0.0 < gamma <= 1.0:
            raise ValueError(f"gamma는 (0, 1] 범위여야 한다: {gamma}")
        self.gamma = gamma
        self._rng = random.Random(seed) if seed is not None else random

    def weights(self, M: int) -> list[float]:
        """w_m = γ^(M−m) — 후반 사이클을 강조한다 (ADR-006).

        m은 0-indexed이며 마지막 사이클(m = M-1)의 가중이 γ로 가장 크다.
        """
        return [self.gamma ** (M - m) for m in range(M)]


@SCHEDULE_REGISTRY.register("fixed")
class FixedSchedule(_BaseSchedule):
    """고정 M. M3의 최소 학습 루프와 Ablation B의 고정 M 스윕에 쓴다."""

    def __init__(self, k: int = 6, gamma: float = 0.85, **_: Any) -> None:
        super().__init__(gamma)
        if k < 1:
            raise ValueError(f"M은 1 이상이어야 한다: {k}")
        self.k = int(k)

    def sample_M(self) -> int:
        return self.k


@SCHEDULE_REGISTRY.register("uniform")
class UniformSchedule(_BaseSchedule):
    """균등 샘플링. 로그정규와의 대조."""

    def __init__(
        self, min: int = 1, max: int = 16, gamma: float = 0.85, seed: Optional[int] = None, **_: Any
    ) -> None:
        super().__init__(gamma, seed)
        self.min, self.max = int(min), int(max)

    def sample_M(self) -> int:
        return self._rng.randint(self.min, self.max)


@SCHEDULE_REGISTRY.register("lognormal")
class LogNormalSchedule(_BaseSchedule):
    """로그정규 샘플링 — 제안서 §5의 기본.

    평균 부근에 질량이 모이되 꼬리가 길어, 대부분의 스텝은 적당한 깊이로 돌면서
    가끔 깊은 전개를 학습한다.

    Args:
        mean: 목표 중앙값 (로그정규의 median = exp(mu)).
        std: 로그 스케일 표준편차.
        max: 상한 (termination.m_max 이하여야 한다).
        min: 하한.
    """

    def __init__(
        self,
        mean: float = 6.0,
        std: float = 0.5,
        max: int = 16,
        min: int = 1,
        gamma: float = 0.85,
        seed: Optional[int] = None,
        **_: Any,
    ) -> None:
        super().__init__(gamma, seed)
        self.mu = math.log(_max(1.0, float(mean)))
        self.sigma = float(std)
        self.max = int(max)
        self.min = int(min)

    def sample_M(self) -> int:
        value = int(round(self._rng.lognormvariate(self.mu, self.sigma)))
        return _min(_max(value, self.min), self.max)



__all__ = ("FixedSchedule", "UniformSchedule", "LogNormalSchedule")
