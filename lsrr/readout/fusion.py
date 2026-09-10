"""어텐션 풀링 융합 (제안서 §4.4).

    α_l = softmax(wᵀ r_l*),  h_SSM = Σ_l α_l r_l*,  h_fusion = h⁽ᴸ⁾ + W_r·h_SSM

`h_ctx`는 **어댑터를 통과하지 않은** 백본 원본 h⁽ᴸ⁾이고, `W_r`은 d_model 공간의
사고 표현을 백본 공간(d_in)으로 되돌린다 (ADR-003).

개작: Legacy_LSRR/lsrr/fusion/attention_pooling.py
"""

from __future__ import annotations

from typing import Any, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from lsrr.core.errors import AssemblyError
from lsrr.core.interfaces import BaseFusionHead
from lsrr.core.registry import FUSION_REGISTRY

FusionType = Literal["residual", "gate", "concat"]


@FUSION_REGISTRY.register("attention_pooling")
class AttentionPoolingFusion(BaseFusionHead):
    """레이어 축 어텐션 풀링 + 잔차 융합.

    Args:
        d_model: 사고 메모리 폭.
        d_out: 출력 폭. **반드시 백본 폭 d_in**이어야 한다 (I8) — 조립 루트가
            `d_out=d_in`으로 넘긴다.
        fusion_type: `residual`(기본) / `gate` / `concat`.
        w_r_init_scale: `W_r` 초기 스케일. 작게 두면 학습 초기 h_fusion ≈ h_ctx가
            되어 판독이 백본의 원래 동작에서 출발하고, 사고 기여분이 점진적으로
            더해진다.
    """

    def __init__(
        self,
        d_model: int = 768,
        d_out: int = 768,
        fusion_type: FusionType = "residual",
        w_r_init_scale: float = 0.01,
        **_: Any,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_out = d_out
        self.fusion_type = fusion_type

        if fusion_type not in ("residual", "gate", "concat"):
            raise AssemblyError(f"fusion_type '{fusion_type}'를 모른다.")

        # α_l = softmax(wᵀ r_l*)
        self.score = nn.Linear(d_model, 1, bias=False)
        # W_r: d_model → d_in
        self.w_r = nn.Linear(d_model, d_out)
        with torch.no_grad():
            self.w_r.weight.mul_(w_r_init_scale)
            self.w_r.bias.zero_()

        if fusion_type == "gate":
            self.gate_proj = nn.Linear(2 * d_out, d_out)
            nn.init.zeros_(self.gate_proj.weight)
            nn.init.constant_(self.gate_proj.bias, -2.0)
        elif fusion_type == "concat":
            self.combine = nn.Linear(2 * d_out, d_out)

        self.last_alpha: Optional[torch.Tensor] = None

    def forward(
        self, R_star: torch.Tensor, h_ctx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """([B, L, d_model], [B, d_in]) → (h_fusion [B, d_in], alpha [B, L])."""
        if h_ctx.shape[-1] != self.d_out:
            raise AssemblyError(
                f"h_ctx 폭({h_ctx.shape[-1]})과 융합 출력 폭({self.d_out})이 다르다. "
                f"h_fusion은 백본 입력 임베딩 공간의 벡터여야 한다 (I8, ADR-003)."
            )

        alpha = F.softmax(self.score(R_star).squeeze(-1), dim=-1)  # [B, L]
        h_ssm = torch.einsum("bl,bld->bd", alpha, R_star)
        projected = self.w_r(h_ssm)

        # α는 손실에 쓰이지 않지만 반드시 기록한다 — analysis/alpha_profile.py가
        # "중간층 집중" 예측(제안서 §7 ii)을 검증하는 데이터다.
        self.last_alpha = alpha.detach()

        if self.fusion_type == "residual":
            h_fusion = h_ctx + projected
        elif self.fusion_type == "gate":
            g = torch.sigmoid(self.gate_proj(torch.cat([h_ctx, projected], dim=-1)))
            h_fusion = (1.0 - g) * h_ctx + g * projected
        else:
            h_fusion = self.combine(torch.cat([h_ctx, projected], dim=-1))

        return h_fusion, alpha

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, d_out={self.d_out}, type={self.fusion_type}"


__all__ = ("AttentionPoolingFusion", "FusionType")
