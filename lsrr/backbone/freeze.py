"""백본 동결과 지문 (I1).

**백본 base 가중치 `W₀` 는 전 학습 과정에서 동결이다.** Phase A 의 학습 대상은
레이어 어댑터 + SSM 엔진 + 풀링/융합 헤드뿐이고(백본 대비 약 3% 이내),
Phase B 는 여기에 LoRA 델타 `ΔW = BA` 가 더해진다 — 델타는 base 텐서를 갱신하지
않는 **별도 파라미터**이므로 `W₀` 불변이라는 주장은 그대로다 (ADR-014 의 I1
재정의). 그래서 지문도 base 만 해싱한다: 델타를 넣으면 지문이 LoRA 학습마다
바뀌어 I1 검증이 무의미해진다.

v2.1 §5.0 이전 문서는 "LoRA 불포함" 으로 적혀 있다. ADR-014 가 그 범위를
"디코딩 전용" 으로 한정하며 뒤집었다.
"""

from __future__ import annotations

from typing import Any, Optional

import torch.nn as nn

from lsrr.core.invariants import assert_frozen, freeze_module, weight_hash

HASH_TENSOR_LIMIT = 8
"""지문에 쓸 텐서 수. 전체 해싱은 대형 백본에서 느리다 — 앞의 몇 개만으로도
가중치 변화 검출에는 충분하고, 정밀 대조가 필요하면 limit=None으로 부른다."""


def freeze_backbone(model: nn.Module) -> nn.Module:
    """동결 + eval 고정 후 즉시 검증한다.

    동결을 '했다'와 '되었다'는 다르다 — 여기서 바로 확인한다.
    """
    freeze_module(model)
    assert_frozen(model, what="backbone")
    return model


def backbone_fingerprint(
    model: nn.Module, limit: Optional[int] = HASH_TENSOR_LIMIT
) -> str:
    """가중치 지문. 학습 전후 대조로 I1을 검증한다 (런 메타에 기록)."""
    return weight_hash(model, num_tensors=limit)


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


__all__ = (
    "HASH_TENSOR_LIMIT",
    "freeze_backbone",
    "backbone_fingerprint",
    "count_parameters",
)
