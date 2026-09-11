"""정제 갱신식의 유일한 소유자 (제안서 §4.2, ADR-017).

    R^(m+1) = Norm( (1-α)·R^(m) + α·S_φ(R^(m), R⁰, m) )

**재귀 적응 장치 3종** (§4.2): 감쇠 갱신(진동 억제), 입력 재주입 R⁰(원 문맥
표류 방지), 사이클 임베딩 m(반복 단계 조건화).

감쇠·재주입·사이클 조건화를 코어가 아니라 래퍼가 하므로, Ablation C(엔진 구조)와
Ablation D(재귀 적응 장치)가 서로 오염되지 않는다 — 코어를 바꿔도 갱신식은
동일하고, 갱신식을 바꿔도 코어는 동일하다.

이식: v1.0:lsrr/engines/wrapper.py
"""

from __future__ import annotations

from typing import Any, Literal, Optional

import torch
import torch.nn as nn

from lsrr.core.interfaces import BaseRefinementEngine
from lsrr.memory.adapters import RMSNorm

ReinjectMode = Literal["none", "add", "concat", "gate"]
StateNorm = Literal["none", "rmsnorm", "layernorm"]


class _ScaleOnly(nn.Module):
    """게인 없는 정규화.

    학습 가능한 이득을 두면 그것이 자라서 폭주가 재현된다 — 코어 진입 전
    정규화가 이미 그렇게 뚫렸다 (F-028). 상태 carry 에는 게인이 없어야 한다.
    """

    def __init__(self, mode: str, eps: float = 1e-6) -> None:
        super().__init__()
        self.mode = mode
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = x.float()
        if self.mode == "layernorm":
            f = f - f.mean(-1, keepdim=True)
        scale = f.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (f * scale).to(x.dtype)

    def extra_repr(self) -> str:
        return f"mode={self.mode}"


class EngineWrapper(BaseRefinementEngine):
    """코어 연산자를 감싸 §4.2 갱신식을 적용한다.

    Args:
        core: `[B, L, d_model] → [B, L, d_model]` 순수 함수.
        d_model: 사고 메모리 폭.
        damping_alpha: 감쇠 계수 α. 1.0이면 감쇠 없음 — Ablation D의 스윕 축이며,
            수렴 강제와 진동형 표현력 사이의 트레이드오프를 정량화한다.
            기본값 0.8은 v2.1 §4.2 (v1은 0.5였다).
        max_cycles: 사이클 임베딩 테이블 크기. `termination.m_max` 이상이어야 한다.
        cycle_embedding: 사이클 인덱스 조건화 여부.
        reinject_r0: R⁰ 재주입 방식.
        norm_type: 코어 진입 전 정규화.
        state_norm: 갱신 **뒤** 상태 정규화 (ADR-017). 코어의 입력은 원래도
            정규화돼 있었지만 carry 되는 상태는 아니었고, 상태 RMS 가 학습
            250스텝 만에 10¹⁴ 배로 갔다 (F-028). 손실은 판독의 RMS 보정 때문에
            `R` 의 상수배에 불변이라 ‖R‖ 을 묶을 기울기 압력이 없다.
            §4.3 의 **고정** ε 은 스케일이 비교 가능할 때만 뜻이 있으므로
            기본은 `rmsnorm` 이다. `none` 이 v2.1 원문 거동이며 Ablation D 축이다.
    """

    def __init__(
        self,
        core: nn.Module,
        d_model: int = 768,
        damping_alpha: float = 0.8,
        max_cycles: int = 32,
        cycle_embedding: bool = True,
        reinject_r0: ReinjectMode = "gate",
        norm_type: str = "rmsnorm",
        state_norm: StateNorm = "rmsnorm",
        **_: Any,
    ) -> None:
        super().__init__()
        if not 0.0 < damping_alpha <= 1.0:
            raise ValueError(f"damping_alpha는 (0, 1] 범위여야 한다: {damping_alpha}")
        if reinject_r0 not in ("none", "add", "concat", "gate"):
            raise ValueError(f"reinject_r0 '{reinject_r0}'를 모른다.")
        if state_norm not in ("none", "rmsnorm", "layernorm"):
            raise ValueError(
                f"engine.state_norm '{state_norm}'를 모른다. "
                f"가능: none/rmsnorm/layernorm (ADR-017)."
            )

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
        self.state_norm_type = state_norm
        self.state_norm = (
            None if state_norm == "none" else _ScaleOnly(state_norm)
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
        R_next = (1.0 - self.damping_alpha) * R_m + self.damping_alpha * f
        # 갱신 **뒤** 정규화 (ADR-017). 없으면 Δ 가 학습 중 자릿수를 바꿔
        # §4.3 의 고정 ε 이 무의미해지고, β = softmax(w_pool·r_l) 가 포화한다.
        return R_next if self.state_norm is None else self.state_norm(R_next)

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, alpha={self.damping_alpha}, "
            f"reinject={self.reinject_r0}, cycle_emb={self.cycle_emb is not None}, "
            f"state_norm={self.state_norm_type}"
        )


__all__ = ("EngineWrapper", "ReinjectMode", "StateNorm")
