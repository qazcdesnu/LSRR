"""레이어 어댑터 — 층간 표현 공간 정렬 (제안서 §4.1).

    "층간 표현 공간은 정렬되어 있지 않으므로(tuned lens가 레이어별 변환기를
     필요로 한 이유; 층별 노름 증가 현상), 레이어별 학습 아핀 어댑터와 정규화,
     레이어 위치 임베딩을 전처리로 둔다."

정렬 없이 레이어 축을 스캔하면 그 축은 의미 있는 시퀀스가 아니다.

이식: v1.0:lsrr/adapters/layer_adapter.py
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn as nn

from lsrr.core.interfaces import BaseLayerAdapter
from lsrr.core.registry import ADAPTER_REGISTRY
from lsrr.memory.layer_embedding import build_layer_embedding


class RMSNorm(nn.Module):
    """층별 노름 증가를 흡수한다.

    LayerNorm과 달리 평균을 빼지 않으므로 잔차 스트림의 방향 정보를 덜 훼손한다.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight


@ADAPTER_REGISTRY.register("per_layer_affine")
class PerLayerAffineAdapter(BaseLayerAdapter):
    """레이어마다 별도 아핀 변환 — 기본 구현.

    파라미터: `L × d_in × d_model`. GPT-2 전층(L=12, 768→768) 기준 약 7.1M으로
    백본 124M의 5.7%다. 제안서 §5의 "3% 이내"를 지키려면 `d_model`을 줄인다
    (ADR-003으로 `d_model`이 `d_in`에서 자유로워졌다).

    Args:
        d_in: 백본 폭.
        d_model: 사고 메모리 폭.
        num_layers: 범위 선택 **후**의 레이어 수 L'.
        layer_pos_emb: 레이어 위치 임베딩 종류 (`learned`/`sinusoidal`/`none`).
        use_rmsnorm: 아핀 후 RMSNorm 적용 여부.
    """

    def __init__(
        self,
        d_in: int = 768,
        d_model: int = 768,
        num_layers: int = 12,
        layer_pos_emb: str | bool = "learned",
        use_rmsnorm: bool = True,
        init_scale: Optional[float] = None,
        **_: Any,
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self._d_model = d_model
        self.num_layers = num_layers

        self.weight = nn.Parameter(torch.empty(num_layers, d_in, d_model))
        self.bias = nn.Parameter(torch.zeros(num_layers, d_model))
        if init_scale is not None:
            nn.init.normal_(self.weight, std=init_scale)
        else:
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        self.norm = RMSNorm(d_model) if use_rmsnorm else nn.Identity()
        self.layer_emb = build_layer_embedding(layer_pos_emb, num_layers, d_model)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """[B, L', d_in] → R0 [B, L', d_model]."""
        L = H.shape[1]
        if L > self.num_layers:
            raise ValueError(
                f"어댑터가 {self.num_layers}개 레이어용으로 만들어졌는데 "
                f"{L}개를 받았다. 범위 선택(scope)과 어댑터가 어긋났다."
            )
        out = torch.einsum("bli,lio->blo", H, self.weight[:L]) + self.bias[:L]
        out = self.norm(out)
        return self.layer_emb(out) if self.layer_emb is not None else out

    @property
    def d_model(self) -> int:
        return self._d_model


@ADAPTER_REGISTRY.register("shared_affine")
class SharedAffineAdapter(BaseLayerAdapter):
    """전 레이어 공유 아핀.

    파라미터가 `d_in × d_model` 하나로 줄어든다. 층간 공간이 실제로 정렬되어
    있다면 이것으로 충분하다 — 즉 `per_layer_affine`과의 차이가 곧 §4.1
    전제(층간 비정렬)의 크기다. 절제 대상.
    """

    def __init__(
        self,
        d_in: int = 768,
        d_model: int = 768,
        num_layers: int = 12,
        layer_pos_emb: str | bool = "learned",
        use_rmsnorm: bool = True,
        **_: Any,
    ) -> None:
        super().__init__()
        self._d_model = d_model
        self.proj = nn.Linear(d_in, d_model)
        self.norm = RMSNorm(d_model) if use_rmsnorm else nn.Identity()
        self.layer_emb = build_layer_embedding(layer_pos_emb, num_layers, d_model)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        out = self.norm(self.proj(H))
        return self.layer_emb(out) if self.layer_emb is not None else out

    @property
    def d_model(self) -> int:
        return self._d_model


@ADAPTER_REGISTRY.register("identity")
class IdentityAdapter(BaseLayerAdapter):
    """정렬을 하지 않는 무력화 구현.

    `d_in != d_model`이면 편향 없는 사영만 둔다. "어댑터가 없을 때"를 같은
    인터페이스로 표현하기 위한 슬롯이다 (CONVENTIONS §1.3).
    """

    def __init__(self, d_in: int = 768, d_model: Optional[int] = None, **_: Any) -> None:
        super().__init__()
        self._d_model = d_model or d_in
        self.proj = (
            nn.Identity() if self._d_model == d_in else nn.Linear(d_in, self._d_model, bias=False)
        )

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        return self.proj(H)

    @property
    def d_model(self) -> int:
        return self._d_model


__all__ = (
    "RMSNorm",
    "PerLayerAffineAdapter",
    "SharedAffineAdapter",
    "IdentityAdapter",
)
