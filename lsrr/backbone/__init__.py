"""동결 백본 — 1회 인코딩, 레이어 추출, 질문 풀링, 연속 디코딩.

이 패키지는 동결된 트랜스포머와 닿는 모든 것을 소유한다. 백본은 **절대**
사이클 루프 안에서 호출되지 않는다 (I2).
"""

from lsrr.backbone.continuation import BackboneContinuation, rewind_cache
from lsrr.backbone.extractor import extract, stack_hidden_states
from lsrr.backbone.freeze import (
    backbone_fingerprint,
    count_parameters,
    freeze_backbone,
)
from lsrr.backbone.pooling import AttentionPooler, MeanPooler
from lsrr.backbone.session import HFFrozenCausalBackbone

__all__ = (
    "HFFrozenCausalBackbone",
    "BackboneContinuation",
    "rewind_cache",
    "AttentionPooler",
    "MeanPooler",
    "freeze_backbone",
    "backbone_fingerprint",
    "count_parameters",
    "stack_hidden_states",
    "extract",
)
