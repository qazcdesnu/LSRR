"""백본 동결과 지문 (I1).

제안서 §5: 백본은 완전 동결하며 LoRA도 쓰지 않는다. 학습 대상은 레이어 어댑터 +
SSM 엔진 + 풀링/융합 헤드뿐이다(백본 대비 약 3% 이내).
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
