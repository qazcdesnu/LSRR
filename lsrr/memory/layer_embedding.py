"""레이어 위치 임베딩 (제안서 §4.1).

레이어 축은 순서가 있는 축이다 — 하위층과 상위층은 역할이 다르다(stages 가설).
SSM 스캔은 순서를 보지만, 어느 위치가 '몇 번째 층'인지는 별도로 알려 줘야 한다.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class LearnedLayerEmbedding(nn.Module):
    """학습형 레이어 위치 임베딩. 파라미터 `L × d_model`."""

    def __init__(self, num_layers: int, d_model: int, init_std: float = 0.02) -> None:
        super().__init__()
        self.emb = nn.Parameter(torch.randn(1, num_layers, d_model) * init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.emb[:, : x.shape[1], :]


class SinusoidalLayerEmbedding(nn.Module):
    """고정 사인형. 학습 파라미터가 없어 예산 정합 비교에서 유용하다."""

    def __init__(self, num_layers: int, d_model: int) -> None:
        super().__init__()
        pos = torch.arange(num_layers, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        emb = torch.zeros(num_layers, d_model)
        emb[:, 0::2] = torch.sin(pos * div)
        emb[:, 1::2] = torch.cos(pos * div)[:, : emb[:, 1::2].shape[1]]
        self.register_buffer("emb", emb.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.emb[:, : x.shape[1], :]


def build_layer_embedding(
    kind: str | bool, num_layers: int, d_model: int
) -> Optional[nn.Module]:
    """설정값 → 임베딩 모듈. `True`는 하위 호환으로 `learned`를 뜻한다."""
    if kind is True:
        kind = "learned"
    elif kind is False or kind is None:
        kind = "none"

    if kind == "none":
        return None
    if kind == "learned":
        return LearnedLayerEmbedding(num_layers, d_model)
    if kind == "sinusoidal":
        return SinusoidalLayerEmbedding(num_layers, d_model)
    raise ValueError(
        f"layer_pos_emb '{kind}'를 모른다. 가능: learned / sinusoidal / none"
    )


__all__ = (
    "LearnedLayerEmbedding",
    "SinusoidalLayerEmbedding",
    "build_layer_embedding",
)
