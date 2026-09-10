"""정제 갱신식의 유일한 소유자 (제안서 §4.2).

    R^(m+1) = (1-α)·R^(m) + α·S_φ(R^(m), R⁰, m)

**재귀 적응 장치 3종** (§4.2): 감쇠 갱신(진동 억제), 입력 재주입 R⁰(원 문맥
표류 방지), 사이클 임베딩 m(반복 단계 조건화).

감쇠·재주입·사이클 조건화를 코어가 아니라 래퍼가 하므로, Ablation C(엔진 구조)와
Ablation D(재귀 적응 장치)가 서로 오염되지 않는다 — 코어를 바꿔도 갱신식은
동일하고, 갱신식을 바꿔도 코어는 동일하다.

이식: Legacy_LSRR/lsrr/engines/wrapper.py
"""

from __future__ import annotations

from typing import Any, Literal, Optional

import torch
import torch.nn as nn

from lsrr.core.interfaces import BaseRefinementEngine
from lsrr.memory.adapters import RMSNorm

ReinjectMode = Literal["none", "add", "concat", "gate"]


class EngineWrapper(BaseRefinementEngine):
    """코어 연산자를 감싸 §4.2 갱신식을 적용한다.

    Args:
        core: `[B, L, d_model] → [B, L, d_model]` 순수 함수.
        d_model: 사고 메모리 폭.
        damping_alpha: 감쇠 계수 α. 1.0이면 감쇠 없음 — Ablation D의 스윕 축이며,
            수렴 강제와 진동형 표현력 사이의 트레이드오프를 정량화한다.
        max_cycles: 사이클 임베딩 테이블 크기. `termination.m_max` 이상이어야 한다.
        cycle_embedding: 사이클 인덱스 조건화 여부.
        reinject_r0: R⁰ 재주입 방식.
        norm_type: 코어 진입 전 정규화.
    """

    def __init__(
        self,
        core: nn.Module,
        d_model: int = 768,
        damping_alpha: float = 0.5,
        max_cycles: int = 32,
        cycle_embedding: bool = True,
        reinject_r0: ReinjectMode = "gate",
        norm_type: str = "rmsnorm",
        **_: Any,
    ) -> None:
        super().__init__()
        if not 0.0 < damping_alpha <= 1.0:
            raise ValueError(f"damping_alpha는 (0, 1] 범위여야 한다: {damping_alpha}")
        if reinject_r0 not in ("none", "add", "concat", "gate"):
            raise ValueError(f"reinject_r0 '{reinject_r0}'를 모른다.")

        self.core = core
        self.d_model = d_model
        self.damping_alpha = damping_alpha
        self.max_cycles = max_cycles
        self.reinject_r0 = reinject_r0

        self.cycle_emb = nn.Embedding(max_cycles, d_model) if cycle_embedding else None
        if self.cycle_emb is not None:
            nn.init.normal_(self.cycle_emb.weight, std=0.02)

        if reinject_r0 == "gate":
            self.gate_proj = nn.Linear(2 * d_model, d_model)
            nn.init.zeros_(self.gate_proj.weight)
            nn.init.constant_(self.gate_proj.bias, -1.0)  # 초기엔 재주입 약하게
        elif reinject_r0 == "concat":
            self.concat_proj = nn.Linear(2 * d_model, d_model)

        self.norm = (
            nn.LayerNorm(d_model) if norm_type == "layernorm" else RMSNorm(d_model)
        )

    def forward_step(
        self, R_m: torch.Tensor, R0: torch.Tensor, m: int
    ) -> torch.Tensor:
        """정제 1사이클. [B, L, d_model] → [B, L, d_model]."""
        x = R_m

        if self.cycle_emb is not None:
            idx = torch.tensor(
                min(m, self.max_cycles - 1), device=R_m.device, dtype=torch.long
            )
            x = x + self.cycle_emb(idx).view(1, 1, -1)

        if self.reinject_r0 == "add":
            x = x + R0
        elif self.reinject_r0 == "concat":
            x = self.concat_proj(torch.cat([x, R0], dim=-1))
        elif self.reinject_r0 == "gate":
            g = torch.sigmoid(self.gate_proj(torch.cat([x, R0], dim=-1)))
            x = (1.0 - g) * x + g * R0

        f = self.core(self.norm(x))
        return (1.0 - self.damping_alpha) * R_m + self.damping_alpha * f

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, alpha={self.damping_alpha}, "
            f"reinject={self.reinject_r0}, cycle_emb={self.cycle_emb is not None}"
        )


__all__ = ("EngineWrapper", "ReinjectMode")
