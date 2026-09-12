"""No-CoT 대조군 — pause 융합 (Coconut 의 Pause token 기준선과 같은 구성).

지금까지 "엔진이 있는 조건끼리" 만 비교했지 "엔진이 없는 것" 과 비교한 적이
없다. 이 헤드가 그 하한을 준다. 고정하는 것:

- 출력이 질문(R_star·h_ctx)에 **전혀** 의존하지 않는다
- 엔진·어댑터에 그래디언트가 흐르지 않는다 — 흐르면 대조군이 아니다
- 상수 벡터 자체는 학습된다
"""

from __future__ import annotations

import torch
import torch.nn as nn

from lsrr.core.registry import FUSION_REGISTRY

B, L, D_MODEL, D_IN = 4, 12, 64, 96


def _head():
    return FUSION_REGISTRY.build({"type": "pause", "d_model": D_MODEL, "d_out": D_IN})


def test_output_ignores_the_question():
    head = _head()
    R1, R2 = torch.randn(B, L, D_MODEL), torch.randn(B, L, D_MODEL)
    c1, c2 = torch.randn(B, D_IN), torch.randn(B, D_IN)
    h1, _ = head(R1, c1); h2, _ = head(R2, c2)
    assert torch.equal(h1, h2)
    assert h1.shape == (B, D_IN)


def test_no_gradient_reaches_the_engine_side():
    """R_star 로 그래디언트가 흐르면 엔진이 학습되고, 그러면 대조군이 아니다."""
    head = _head()
    R = torch.randn(B, L, D_MODEL, requires_grad=True)
    c = torch.randn(B, D_IN, requires_grad=True)
    h, _ = head(R, c)
    h.sum().backward()
    assert R.grad is None and c.grad is None
    assert head.vector.grad is not None


def test_only_the_vector_is_trainable():
    head = _head()
    names = [n for n, p in head.named_parameters() if p.requires_grad]
    assert names == ["vector"]
    assert head.vector.numel() == D_IN
