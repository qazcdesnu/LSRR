"""문맥 구성 — 마지막 토큰 열 + 질문 풀링 열 (제안서 §4.1, ADR-004)."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from lsrr.core.interfaces import BaseMemoryComposer
from lsrr.core.registry import COMPOSER_REGISTRY


@COMPOSER_REGISTRY.register("last_only")
class LastOnlyComposer(BaseMemoryComposer):
    """보완항을 쓰지 않는 무력화 구현 — ADR-004의 대조 조건."""

    def __init__(self, **_: Any) -> None:
        super().__init__()

    def forward(
        self, H_last: torch.Tensor, H_pool: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return H_last


@COMPOSER_REGISTRY.register("add")
class AddComposer(BaseMemoryComposer):
    """단순 합. 학습 파라미터가 없다."""

    def __init__(self, pool_weight: float = 0.5, **_: Any) -> None:
        super().__init__()
        self.pool_weight = pool_weight

    def forward(
        self, H_last: torch.Tensor, H_pool: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if H_pool is None:
            return H_last
        return H_last + self.pool_weight * H_pool


@COMPOSER_REGISTRY.register("gate")
class GateComposer(BaseMemoryComposer):
    """레이어별 게이트 결합 — 기본 구현.

        g = sigmoid(W [H_last ; H_pool]),  H = (1-g)·H_last + g·H_pool

    게이트를 0으로 초기화(편향 `gate_init`)해 학습 초기에는 `H_last` 단독에서
    출발하고 보완항이 점진적으로 섞이게 한다. 보완항이 쓸모없다면 게이트가
    닫힌 채로 남으므로, 학습된 게이트 값 자체가 ADR-004의 사후 증거가 된다.
    """

    def __init__(
        self, d_in: int = 768, gate_init: float = -2.0, **_: Any
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(2 * d_in, d_in)
        nn.init.zeros_(self.proj.weight)
        nn.init.constant_(self.proj.bias, gate_init)
        self.last_gate: Optional[torch.Tensor] = None

    def forward(
        self, H_last: torch.Tensor, H_pool: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if H_pool is None:
            return H_last
        g = torch.sigmoid(self.proj(torch.cat([H_last, H_pool], dim=-1)))
        self.last_gate = g.detach().mean().reshape(())  # 진단: 보완항 사용량
        return (1.0 - g) * H_last + g * H_pool


@COMPOSER_REGISTRY.register("concat")
class ConcatComposer(BaseMemoryComposer):
    """이어 붙인 뒤 사영. 표현력은 가장 크지만 파라미터도 가장 많다."""

    def __init__(self, d_in: int = 768, **_: Any) -> None:
        super().__init__()
        self.proj = nn.Linear(2 * d_in, d_in)

    def forward(
        self, H_last: torch.Tensor, H_pool: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if H_pool is None:
            return H_last
        return self.proj(torch.cat([H_last, H_pool], dim=-1))


__all__ = ("LastOnlyComposer", "AddComposer", "GateComposer", "ConcatComposer")
