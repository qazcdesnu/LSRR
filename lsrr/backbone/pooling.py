"""레이어별 질문 전체 풀링 — 글로벌 문맥 보완항 (제안서 §4.1, ADR-004).

    "단일 토큰의 정보 병목을 막기 위해 질문 전체의 어텐션 풀링 결과를 결합해
     글로벌 문맥을 보완한다."

레거시는 `last_token`과 `mean_pooling` 중 **택일**만 지원했고 결합 경로가 없었다.
L≈32 길이의 레이어 축 메모리 전체가 단일 토큰 위치에서 온다면, 다중 홉 과제에서
문맥의 다른 위치에 흩어진 브리지 정보가 애초에 메모리에 들어오지 못한다.

**소유권 주의:** 이 모듈들은 학습 대상이다(제안서 §5의 "풀링/융합 헤드"). 동결
백본 세션 안이 아니라 모델의 자식으로 등록된다 (ADR-012).
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from lsrr.core.interfaces import BaseContextPooler
from lsrr.core.registry import POOLER_REGISTRY


def _masked_fill_scores(
    scores: torch.Tensor, attention_mask: Optional[torch.Tensor]
) -> torch.Tensor:
    """패딩 위치를 -inf로 만들어 softmax에서 배제한다. [B, L, T]."""
    if attention_mask is None:
        return scores
    mask = attention_mask[:, None, :].to(torch.bool)
    return scores.masked_fill(~mask, torch.finfo(scores.dtype).min)


@POOLER_REGISTRY.register("attn_pool")
class AttentionPooler(BaseContextPooler):
    """레이어별 학습 쿼리로 질문 토큰에 어텐션 풀링한다.

    레이어마다 별도 쿼리를 두는 것이 기본이다 — 층간 표현 공간이 정렬되어
    있지 않으므로(제안서 §4.1) 공유 쿼리는 층마다 다른 것을 고른다.

    파라미터: num_layers × d_in (GPT-2 기준 12×768 ≈ 9.2K, 무시할 수준).
    """

    def __init__(
        self,
        d_in: int = 768,
        num_layers: int = 12,
        per_layer_query: bool = True,
        temperature: float = 1.0,
        match_norm: bool = True,
        **_: Any,
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.num_layers = num_layers
        self.per_layer_query = per_layer_query
        self.temperature = temperature
        # 풀링 출력의 노름을 마지막 토큰 열에 맞춘다. 맞추지 않으면 학습 중
        # 쿼리가 커지면서 풀링 항의 노름이 폭주하고(M3 실측: H_last의 15배),
        # 결합 게이트가 열리는 순간 보완항이 본 신호를 덮어쓴다 (FINDINGS F-011).
        self.match_norm = match_norm
        shape = (num_layers, d_in) if per_layer_query else (1, d_in)
        self.query = nn.Parameter(torch.randn(*shape) * (d_in**-0.5))
        self.last_alpha: Optional[torch.Tensor] = None

    def forward(
        self,
        hidden_stack: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """[B, L, T, d] → [B, L, d]."""
        L = hidden_stack.shape[1]
        q = self.query[:L] if self.per_layer_query else self.query.expand(L, -1)
        scores = torch.einsum("bltd,ld->blt", hidden_stack, q) / self.temperature
        scores = _masked_fill_scores(scores, attention_mask)
        alpha = torch.softmax(scores, dim=-1)
        self.last_alpha = alpha.detach()  # 분석용 (어디를 봤는가)
        pooled = torch.einsum("blt,bltd->bld", alpha, hidden_stack)

        if self.match_norm:
            ref = hidden_stack[:, :, -1, :].norm(dim=-1, keepdim=True)
            pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-6) * ref
        return pooled


@POOLER_REGISTRY.register("mean_pool")
class MeanPooler(BaseContextPooler):
    """마스크 가중 평균. 학습 파라미터가 없는 대조 조건."""

    def __init__(self, **_: Any) -> None:
        super().__init__()

    def forward(
        self,
        hidden_stack: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if attention_mask is None:
            return hidden_stack.mean(dim=2)
        w = attention_mask[:, None, :, None].to(hidden_stack.dtype)
        return (hidden_stack * w).sum(dim=2) / w.sum(dim=2).clamp(min=1.0)


__all__ = ("AttentionPooler", "MeanPooler")
