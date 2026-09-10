"""메모리 파이프라인 — ContextBundle → R⁰.

순서는 고정이다: **compose → scope → adapt**.

scope가 adapt보다 먼저인 이유: 선택되지 않은 레이어에 어댑터 파라미터를
할당하지 않기 위해서다. Ablation A의 각 조건이 서로 다른 파라미터 수를 갖게
되므로, 비교 시 이 점을 보고에 명시한다.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from lsrr.core.types import ContextBundle


class LayerMemoryPipeline(nn.Module):
    """구성·범위·어댑터를 하나의 모듈로 묶는다.

    조립 루트가 슬롯을 골라 넘기고, 여기서는 순서만 보장한다.
    """

    def __init__(
        self,
        composer: Optional[nn.Module] = None,
        scope: Optional[nn.Module] = None,
        adapter: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.composer = composer
        self.scope = scope
        self.adapter = adapter

    @property
    def d_model(self) -> Optional[int]:
        return getattr(self.adapter, "d_model", None)

    def layer_indices(self, num_layers: int) -> list[int]:
        """원본 백본 레이어 인덱스. 분석에서 층별 귀속에 쓴다."""
        if self.scope is None:
            return list(range(num_layers))
        return self.scope.layer_indices(num_layers)

    def forward(self, context: ContextBundle) -> torch.Tensor:
        """ContextBundle → R⁰ [B, L', d_model]."""
        H = context.H_last
        if self.composer is not None:
            H = self.composer(H, context.H_pool)
        if self.scope is not None:
            H = self.scope(H)
        if self.adapter is not None:
            H = self.adapter(H)
        return H

    def diagnostics(self) -> dict[str, float]:
        """구성 게이트 값 등 진단 지표 (ADR-004 사후 증거)."""
        out: dict[str, float] = {}
        gate = getattr(self.composer, "last_gate", None)
        if gate is not None:
            out["composer_gate"] = float(gate)
        return out


__all__ = ("LayerMemoryPipeline",)
