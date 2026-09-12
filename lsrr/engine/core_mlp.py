"""1회 통과 MLP — Phase 0 게이트 ③의 비교 대상 (제안서 §6.3).

    "③ H+MLP 1회 통과 대비 유의한 우위. ③이 실패하면 반복의 기여가 없다는
     뜻이므로 아키텍처를 재검토한다(킬 스위치)."

믹서 행렬 클래스 관점에서 이것은 **대각 행렬**이다 — 레이어 간 정보를 전혀
섞지 않는다. 따라서 "레이어 축을 섞는 것이 도움이 되는가"의 하한이다.

이식: v1.0:lsrr/engines/mlp_onepass.py
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from lsrr.core.registry import ENGINE_REGISTRY
from lsrr.engine.wrapper import EngineWrapper


class MLPCore(nn.Module):
    """레이어별 독립 MLP. 레이어 축을 섞지 않는다."""

    def __init__(
        self,
        d_model: int = 768,
        expansion: int = 2,
        n_blocks: int = 1,
        dropout: float = 0.0,
        **_: Any,
    ) -> None:
        super().__init__()
        hidden = d_model * expansion
        blocks = []
        for _i in range(n_blocks):
            blocks.append(
                nn.Sequential(
                    nn.Linear(d_model, hidden),
                    nn.GELU(),
                    nn.Dropout(dropout) if dropout else nn.Identity(),
                    nn.Linear(hidden, d_model),
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = x + blk(x)
        return x


@ENGINE_REGISTRY.register("mlp_onepass")
def build_mlp_onepass(
    d_model: int = 768,
    expansion: int = 2,
    n_blocks: int = 1,
    dropout: float = 0.0,
    damping_alpha: float = 0.8,
    max_cycles: int = 32,
    cycle_embedding: bool = True,
    reinject_r0: str = "gate",
    norm_type: str = "rmsnorm",
    state_norm: str = "rmsnorm",
    mix_norm: bool = False,
    **kwargs: Any,
) -> EngineWrapper:
    """레지스트리 진입점. 코어를 래퍼로 감싸 §4.2 갱신식을 적용한다.

    "1회 통과"는 **사이클을 1회만 돌린다**는 뜻이지 래퍼를 벗긴다는 뜻이 아니다.
    반복 여부는 `recurrence`가 정하며, 여기서 엔진 구조만 다르게 한다 —
    그래야 게이트 ③이 "반복의 기여"만을 분리해 측정한다.
    """
    core = MLPCore(
        d_model=d_model, expansion=expansion, n_blocks=n_blocks, dropout=dropout
    )
    return EngineWrapper(
        core=core,
        d_model=d_model,
        damping_alpha=damping_alpha,
        max_cycles=max_cycles,
        cycle_embedding=cycle_embedding,
        reinject_r0=reinject_r0,
        norm_type=norm_type,
        state_norm=state_norm,
        mix_norm=mix_norm,
    )


__all__ = ("MLPCore", "build_mlp_onepass")
