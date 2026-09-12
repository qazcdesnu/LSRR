"""어텐션 풀링 융합 (제안서 §4.4).

    α_l = softmax(wᵀ r_l*),  h_SSM = Σ_l α_l r_l*,  h_fusion = h⁽ᴸ⁾ + W_r·h_SSM

`h_ctx`는 **어댑터를 통과하지 않은** 백본 원본 h⁽ᴸ⁾이고, `W_r`은 d_model 공간의
사고 표현을 백본 공간(d_in)으로 되돌린다 (ADR-003).

개작: v1.0:lsrr/fusion/attention_pooling.py
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

#: 잔차 앵커 `h_ctx` 를 방출 토큰에 섞는 방식 (ADR-015 / F-025).
#:
#: v1 의 `h = h_ctx + W_r·h_ssm` 에서 앵커는 세 가지 일을 했다.
#:   ① 정보 보험 — 잠재 벡터가 백본에 가는 **유일한** 통로였으므로, 정제가
#:      실패해도 질문 요약이 실려 있으면 최소 성능이 보장됐다.
#:   ② 매니폴드 착지 — 백본 자신이 만든 벡터를 기저로 깔아 분포 근처에 둔다.
#:   ③ 안정화 램프 — W_r 을 작게 시작하면 초기 주입 ≈ h_ctx 라 백본을 놀라게
#:      하지 않고 사고 항이 점진적으로 켜진다 (ResNet 잔차 가지의 논리).
#:
#: **v2 에서 ①과 ②가 소멸한다.** 백본이 질문 전체의 base KV 위에서 디코딩하므로
#: `h_ctx` 가 요약하는 내용은 어텐션이 원본에서 언제든 꺼낼 수 있다 — 잠재 토큰에
#: 또 실으면 순수 중복이다(①). 그리고 Phase B 의 LoRA 수신 정렬이 있으므로 주입
#: 벡터를 `h_ctx` 근처로 위장할 필요가 없다(②). 애초에 노름 232 짜리 `h⁽ᴸ⁾` 은
#: 임베딩 분포(노름 수 단위)와 동떨어져 있어 ② 의 명분이 실측과 맞지 않았다.
#:
#: 남는 ③ 은 상수 앵커 없이도 된다 — 토큰별 RMS 보정(ADR-013)과 W_r 초기화가
#: 이미 제공한다.
#:
#: 그리고 앵커를 **모든** 사이클에 걸면 공통 스탬프가 되어 방출 토큰의 분화를
#: 원천에서 막는다 (F-025: `h⁽¹⁾` vs `h⁽⁵⁾` 코사인 1.0000).
#:
#: **기본값은 `first` 다 (F-027).** 앵커를 전부 빼면 1자리처럼 쉬운 과제에서는
#: 최선이지만(F-026), 2자리에서는 백본이 읽을 출발점이 없어 주입 통로가 늦게
#: 열린다. 첫 토큰에만 두는 것이 두 과제 모두에서 셔플 개입 Δ 가 가장 크다
#: (2자리 10k 기준 4.03 vs 없음 2.76 vs 모든토큰 0.0008).
AnchorMode = Literal["ctx", "none", "ramp", "first"]


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
        anchor: AnchorMode = "first",
        ramp_init: float = -2.0,
        **_: Any,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_out = d_out
        self.fusion_type = fusion_type
        self.anchor = anchor

        if fusion_type not in ("residual", "gate", "concat"):
            raise AssemblyError(f"fusion_type '{fusion_type}'를 모른다.")
        if anchor not in ("ctx", "none", "ramp", "first"):
            raise AssemblyError(f"anchor '{anchor}'를 모른다.")
        if anchor == "ramp":
            # σ(g)·W_r·h_ssm — 앵커 없이 안정화 램프만 남긴다. g 초기값이 작아
            # 학습 초기 주입이 작게 시작한다.
            self.ramp = nn.Parameter(torch.tensor(float(ramp_init)))

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
        self,
        R_star: torch.Tensor,
        h_ctx: torch.Tensor,
        use_anchor: Optional[bool] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """([B, L, d_model], [B, d_in]) → (h_fusion [B, d_in], alpha [B, L]).

        Args:
            use_anchor: `anchor="first"` 에서 호출자가 사이클별로 지정한다.
                None 이면 `anchor` 모드의 기본 거동을 따른다.
        """
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

        anchored = self._anchor_active(use_anchor)
        if not anchored:
            # 순수 사고 항. 토큰별 RMS 보정(ADR-013)이 스케일을 맡는다.
            # `ramp` 게이트는 여기서 걸지 않는다 — RMSNorm 이 스케일 불변이라
            # σ(g) 가 정확히 소거된다(실측 차이 3e-08). 보정 **뒤**에 걸어야
            # 의미가 있으므로 `post_gate()` 가 담당한다.
            return projected, alpha

        if self.fusion_type == "residual":
            h_fusion = h_ctx + projected
        elif self.fusion_type == "gate":
            g = torch.sigmoid(self.gate_proj(torch.cat([h_ctx, projected], dim=-1)))
            h_fusion = (1.0 - g) * h_ctx + g * projected
        else:
            h_fusion = self.combine(torch.cat([h_ctx, projected], dim=-1))

        return h_fusion, alpha

    def post_gate(self, h: torch.Tensor) -> torch.Tensor:
        """보정 **뒤**에 적용하는 안정화 램프. `anchor="ramp"` 에서만 동작한다.

        앵커를 뺀 대신 "작게 시작해 점진적으로 켜진다"는 성질(역할 ③)만 남기는
        장치다. 보정 앞에 걸면 RMS 정규화가 소거하므로 반드시 뒤에 건다.
        """
        if self.anchor != "ramp":
            return h
        return torch.sigmoid(self.ramp) * h

    def _anchor_active(self, use_anchor: Optional[bool]) -> bool:
        """이 호출에서 앵커를 섞을지."""
        if self.anchor == "ctx":
            return True
        if self.anchor in ("none", "ramp"):
            return False
        # anchor == "first": 호출자가 지정한다. 지정이 없으면 켠다(단일 판독 = 첫 토큰).
        return True if use_anchor is None else bool(use_anchor)

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, d_out={self.d_out}, "
            f"type={self.fusion_type}, anchor={self.anchor}"
        )


@FUSION_REGISTRY.register("pause")
class PauseFusion(BaseFusionHead):
    """**No-CoT 대조군** — 질문과 무관한 학습 상수 벡터 하나를 주입한다.

    Coconut 의 *Pause token* 기준선(Goyal et al. 2023; Coconut Table 1 에서
    ProsQA 75.9%)과 같은 구성이다. `R_star` 와 `h_ctx` 를 **모두 무시**한다 —
    엔진·어댑터에는 그래디언트가 흐르지 않으므로 그쪽은 학습되지 않는다.

    왜 "아무것도 주입하지 않는 것" 이 아니라 상수 하나인가: LoRA 는 디코딩
    전용이라(I9) 질문 **뒤** 위치에만 걸린다. 사고 토큰이 0개면 답의 첫 토큰을
    예측하는 위치가 질문 마지막 토큰 = base 가 되어 LoRA 가 아무 일도 못 한다.
    상수 하나를 두면 LoRA 를 타는 위치가 하나 생기고, 엔진 조건과의 차이는
    정확히 "엔진이 만든 슬롯 vs 내용 없는 슬롯" 이 된다.

    `emission: single` 과 함께 쓴다 — 궤적이면 같은 상수가 M번 반복될 뿐이다.
    """

    def __init__(self, d_model: int = 768, d_out: int = 768, **_: Any) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_out = d_out
        self.anchor = "none"
        self.vector = nn.Parameter(torch.zeros(d_out))
        nn.init.normal_(self.vector, std=0.02)
        self.last_alpha: Optional[torch.Tensor] = None

    def forward(
        self,
        R_star: torch.Tensor,
        h_ctx: torch.Tensor,
        use_anchor: Optional[bool] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, L = R_star.shape[0], R_star.shape[1]
        h = self.vector.unsqueeze(0).expand(B, -1)
        alpha = torch.full((B, L), 1.0 / L, device=R_star.device, dtype=R_star.dtype)
        self.last_alpha = alpha
        return h, alpha


__all__ = ("AttentionPoolingFusion", "PauseFusion", "AnchorMode", "FusionType")
