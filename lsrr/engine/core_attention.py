"""동예산 어텐션 블록 — Ablation C의 dense 조건 (제안서 §7).

믹서 행렬 클래스로 보면 **dense**다. 레이어 축 12개에 대한 전연결 믹싱이므로
표현력 상한이지만, 길이에 대해 이차 비용이고 상태 압축이 없다.

이 블록이 존재하는 이유는 "SSM이 필요한가, 그냥 어텐션이면 되지 않는가"를 막기
위해서다. 따라서 **파라미터 예산을 기준 엔진에 맞추는 것이 핵심**이며(ADR-009),
`budget.py`가 FFN 폭을 역산해 그 정합을 보증한다.

이식: Legacy_LSRR/lsrr/engines/attn_block.py
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from lsrr.core.registry import ENGINE_REGISTRY
from lsrr.engine.budget import (
    assert_budget_match,
    solve_attention_d_ffn,
    target_core_params,
)
from lsrr.engine.wrapper import EngineWrapper


class AttentionCore(nn.Module):
    """pre-norm 멀티헤드 자기어텐션 + FFN. `[B, L, d_model] → [B, L, d_model]`.

    레이어 축에는 인과 마스킹을 걸지 않는다 — 완성된 궤적이므로 모든 레이어가
    서로를 볼 수 있다(`core_hydra`의 양방향 근거와 같다).
    """

    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 8,
        d_ffn: Optional[int] = None,
        dropout: float = 0.0,
        **_: Any,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ffn = 4 * d_model if d_ffn is None else int(d_ffn)

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, self.d_ffn),
            nn.GELU(),
            nn.Linear(self.d_ffn, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`[B, L, d_model] → [B, L, d_model]`."""
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        return x + self.ffn(self.norm2(x))


@ENGINE_REGISTRY.register("attn_block")
def build_attn_block(
    d_model: int = 768,
    n_heads: int = 8,
    d_ffn: Optional[int] = None,
    dropout: float = 0.0,
    n_blocks: int = 2,
    budget_match: Optional[str] = "hydra_qs",
    d_state: int = 16,
    expand: int = 2,
    damping_alpha: float = 0.8,
    max_cycles: int = 32,
    cycle_embedding: bool = True,
    reinject_r0: str = "gate",
    norm_type: str = "rmsnorm",
    **_: Any,
) -> EngineWrapper:
    """레지스트리 진입점.

    `d_ffn`을 명시하지 않으면 `budget_match` 기준 엔진의 코어 파라미터 수에
    맞춰 역산한다. 명시하면 그 값을 쓰되 정합은 여전히 검사한다 — 예산이
    어긋난 채로 Ablation C를 돌리면 결론이 무의미하기 때문이다.
    """
    target = None
    if budget_match is not None:
        target = target_core_params(
            budget_match, d_model=d_model, d_state=d_state, expand=expand
        )
        if d_ffn is None:
            d_ffn = solve_attention_d_ffn(target, d_model, n_heads=n_heads)

    cores = [
        AttentionCore(d_model=d_model, n_heads=n_heads, d_ffn=d_ffn, dropout=dropout)
        for _ in range(n_blocks)
    ]
    if target is not None:
        assert_budget_match(cores[0], target, "attn_block", budget_match)

    core: nn.Module = nn.Sequential(*cores) if n_blocks > 1 else cores[0]
    return EngineWrapper(
        core=core,
        d_model=d_model,
        damping_alpha=damping_alpha,
        max_cycles=max_cycles,
        cycle_embedding=cycle_embedding,
        reinject_r0=reinject_r0,
        norm_type=norm_type,
    )


__all__ = ("AttentionCore", "build_attn_block")
