"""메모리 파이프라인 — ContextBundle → R⁰.

순서는 고정이다: **center → compose → scope → adapt**.

center (ADR-019): 백본 층 상태에서 **학습셋 평균**을 뺀다. 마지막 질문 토큰의
층 상태는 샘플 간 코사인 0.9755 로 거의 같은 방향이고(F-030), 그 공유 성분은
고정 벡터 하나라 학습셋 평균을 빼면 0.0005 가 된다(F-040). 특징축 정규화
(LayerNorm·RMSNorm)로는 지워지지 않는다 — 지워야 할 것은 샘플 축의 평균이다.
학습 가능한 오프셋으로 두지 않는다: 손실이 답만 보는 한 공유 성분을 지울
압력이 없어 원위치한다(어댑터가 0.974 → 0.995 로 오히려 키웠다).

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
        center: str = "none",
        d_in: Optional[int] = None,
        num_layers: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.composer = composer
        self.scope = scope
        self.adapter = adapter
        if center not in ("none", "train_mean"):
            raise ValueError(f"memory.center '{center}' 를 모른다. 가능: none / train_mean (ADR-019).")
        self.center = center
        if center == "train_mean":
            if d_in is None or num_layers is None:
                raise ValueError("memory.center=train_mean 은 d_in 과 num_layers 가 필요하다.")
            # 비학습 버퍼 — 체크포인트에 포함되고, 추론 시 배치와 무관하다.
            self.register_buffer("mu_last", torch.zeros(num_layers, d_in))
            self.register_buffer("mu_pool", torch.zeros(num_layers, d_in))
            self.register_buffer("center_ready", torch.zeros((), dtype=torch.bool))

    # ------------------------------------------------------------ center

    @torch.no_grad()
    def estimate_center(
        self, H_last: torch.Tensor, H_pool: Optional[torch.Tensor] = None
    ) -> dict[str, float]:
        """학습셋 층 상태의 평균을 추정해 버퍼에 넣는다 (한 번, 학습 전).

        Args:
            H_last: [N, L, d_in] 학습 표본의 마지막 토큰 층 상태.
            H_pool: [N, L, d_in] 같은 표본의 풀링 상태. 없으면 0 으로 둔다 —
                컴포저가 H_pool 을 쓰면 공유 성분이 그쪽으로 되돌아오므로,
                풀러가 있는 구성에서는 반드시 같이 넘긴다.

        Returns:
            추정 전후의 샘플 간 평균 코사인 — 기록용 (F-040 의 0.9755 → 0.0005).
        """
        if self.center != "train_mean":
            raise RuntimeError("center=train_mean 이 아닌 파이프라인에는 추정할 것이 없다.")
        self.mu_last.copy_(H_last.float().mean(0).to(self.mu_last.dtype))
        if H_pool is not None:
            self.mu_pool.copy_(H_pool.float().mean(0).to(self.mu_pool.dtype))
        self.center_ready.fill_(True)
        return {
            "cosine_before": _mean_cosine(H_last),
            "cosine_after": _mean_cosine(H_last - self.mu_last.to(H_last.device)),
            "num_samples": float(H_last.shape[0]),
        }

    def _apply_center(self, context: ContextBundle) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        H, P = context.H_last, context.H_pool
        if self.center != "train_mean":
            return H, P
        if not bool(self.center_ready):
            raise RuntimeError(
                "memory.center=train_mean 인데 평균이 추정되지 않았다. 학습 전에 "
                "`estimate_center` 를 부르거나, 그것이 담긴 체크포인트를 적재하라 (ADR-019)."
            )
        H = H - self.mu_last.to(H.dtype)
        if P is not None:
            P = P - self.mu_pool.to(P.dtype)
        return H, P

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
        H, P = self._apply_center(context)
        if self.composer is not None:
            H = self.composer(H, P)
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


def _mean_cosine(X: torch.Tensor) -> float:
    """샘플 간 평균 코사인 (L×d 를 펼쳐서). 진단용."""
    n = torch.nn.functional.normalize(X.float().reshape(X.shape[0], -1), dim=-1)
    S = n @ n.T
    off = ~torch.eye(len(n), dtype=torch.bool, device=S.device)
    return float(S[off].mean()) if len(n) > 1 else 1.0


__all__ = ("LayerMemoryPipeline",)
