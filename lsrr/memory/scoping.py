"""레이어 범위 선택 — Ablation A (제안서 §7).

    "h(L) 단독 vs 전체 H vs 중간 구간 vs 후반 구간 — 레이어 축 전개가 사고
     메모리로서 효용이 있음을 증명. 구간 절제는 층별 효용 불균등 가설(stages)의
     검증을 겸함"

구간 경계는 레이어 수의 **비율**로 지정한다. 절대 인덱스로 두면 백본을 바꿀 때
의미가 달라진다.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch

from lsrr.core.interfaces import BaseMemoryScope
from lsrr.core.registry import SCOPE_REGISTRY


class _IndexScope(BaseMemoryScope):
    """인덱스 목록으로 레이어를 고르는 공통 기반."""

    def __init__(self, num_layers: int) -> None:
        super().__init__()
        self.num_layers = int(num_layers)
        idx = torch.tensor(self.layer_indices(self.num_layers), dtype=torch.long)
        self.register_buffer("index", idx, persistent=False)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        return H.index_select(dim=1, index=self.index.to(H.device))


@SCOPE_REGISTRY.register("all_layers")
class AllLayersScope(_IndexScope):
    """전체 레이어 — 기준 조건."""

    def __init__(self, num_layers: int = 12, **_: Any) -> None:
        super().__init__(num_layers)

    def layer_indices(self, num_layers: int) -> list[int]:
        return list(range(num_layers))


@SCOPE_REGISTRY.register("final_only")
class FinalOnlyScope(_IndexScope):
    """h(L) 단독.

    기존 잠재 추론 연구들이 쓰는 재료다(제안서 §2). 레이어 축이 길이 1로 축퇴해
    엔진의 스캔이 무의미해지므로, 이 조건에서의 성능이 곧 "레이어 축 전개를
    쓰지 않았을 때"의 기준선이다.
    """

    def __init__(self, num_layers: int = 12, **_: Any) -> None:
        super().__init__(num_layers)

    def layer_indices(self, num_layers: int) -> list[int]:
        return [num_layers - 1]


class _BandScope(_IndexScope):
    """비율 구간 [lo, hi)로 레이어를 자른다."""

    band: tuple[float, float] = (0.0, 1.0)

    def __init__(self, num_layers: int = 12, band: Sequence[float] | None = None, **_: Any) -> None:
        if band is not None:
            lo, hi = float(band[0]), float(band[1])
            if not 0.0 <= lo < hi <= 1.0:
                raise ValueError(f"구간은 0 <= lo < hi <= 1이어야 한다: {band}")
            self.band = (lo, hi)
        super().__init__(num_layers)

    def layer_indices(self, num_layers: int) -> list[int]:
        lo = int(round(self.band[0] * num_layers))
        hi = int(round(self.band[1] * num_layers))
        hi = max(hi, lo + 1)  # 최소 1개는 남긴다
        return list(range(lo, min(hi, num_layers)))


@SCOPE_REGISTRY.register("mid_band")
class MidBandScope(_BandScope):
    """중간 구간. 다중 홉 브리지 엔티티가 해소되는 곳이라는 가설 (제안서 §3 ①)."""

    band = (0.33, 0.66)


@SCOPE_REGISTRY.register("late_band")
class LateBandScope(_BandScope):
    """후반 구간. 잔차 선명화·정보 삭제 구간이라는 가설."""

    band = (0.66, 1.0)


__all__ = (
    "AllLayersScope",
    "FinalOnlyScope",
    "MidBandScope",
    "LateBandScope",
)
