"""I1 — 백본 **base** 가중치는 동결이며 학습 전후 변하지 않는다.

제안서 §5 는 학습 대상을 레이어 어댑터 + SSM 엔진 + 풀링/융합 헤드로 둔다
(백본 대비 약 3% 이내). v2.1 §5.0 이 Phase B 를 도입하면서 ADR-014 가 I1 을
재정의했다: **base 가중치 `W₀` 는 여전히 불변**이고, LoRA 델타 `ΔW = BA` 는
base 텐서를 갱신하지 않는 별도 파라미터로서 Phase B 에서만 학습된다.

즉 이 연구의 주장("동결 백본에 사후 장착")은 그대로다 — LoRA 는 백본을 바꾸는
것이 아니라 수신기를 덧붙이는 것이다.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from lsrr.backbone.freeze import backbone_fingerprint, count_parameters, freeze_backbone
from lsrr.core.errors import FrozenBackboneViolation
from lsrr.core.invariants import assert_frozen, assert_weights_unchanged

pytestmark = pytest.mark.contract


def test_unfrozen_module_rejected():
    with pytest.raises(FrozenBackboneViolation, match="동결이다"):
        assert_frozen(nn.Linear(4, 4))


def test_lora_delta_is_rejected_unless_explicitly_allowed():
    """Phase A 에 어댑터가 실수로 붙는 것을 잡는다 — 기본은 여전히 엄격이다."""

    class _WithLora(nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = nn.Parameter(torch.zeros(2, 2))
            self.weight = nn.Parameter(torch.zeros(2, 2), requires_grad=False)

    m = _WithLora()
    with pytest.raises(FrozenBackboneViolation, match="ADR-014"):
        assert_frozen(m)
    assert_frozen(m, allow_lora=True)  # Phase B 에서는 정상


def test_fingerprint_ignores_lora_delta():
    """지문에 델타를 넣으면 LoRA 학습마다 바뀌어 I1 검증이 무의미해진다."""
    from lsrr.core.invariants import weight_hash

    class _Base(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(2, 2))

    m = _Base()
    before = weight_hash(m)
    m.register_parameter("lora_A", nn.Parameter(torch.randn(2, 2)))
    assert weight_hash(m) == before
    with torch.no_grad():
        m.lora_A.add_(5.0)
    assert weight_hash(m) == before


def test_freeze_then_verify():
    m = nn.Linear(4, 4)
    freeze_backbone(m)
    assert_frozen(m)
    assert not m.training


def test_changed_weights_detected():
    m = nn.Linear(4, 4)
    before = backbone_fingerprint(m)
    with torch.no_grad():
        m.weight.add_(1.0)
    with pytest.raises(FrozenBackboneViolation, match="가중치가 변했다"):
        assert_weights_unchanged(before, backbone_fingerprint(m))


def test_gpt2_session_is_frozen(gpt2_backbone):
    assert_frozen(gpt2_backbone.model, what="gpt2")
    counts = count_parameters(gpt2_backbone.model)
    assert counts["trainable"] == 0
    assert counts["total"] == gpt2_backbone.num_parameters()


def test_gpt2_weights_survive_a_backward(gpt2_backbone, gpt2_batch, calibrator):
    """실제 역전파를 통과시킨 뒤에도 백본 가중치가 그대로다."""
    before = gpt2_backbone.weight_hash()
    ctx = gpt2_backbone.encode(gpt2_batch["input_ids"], gpt2_batch["attention_mask"])
    h = calibrator(ctx.h_ctx.clone().requires_grad_(True))
    logits = gpt2_backbone.answer_head.teacher_forced(
        h, ctx.kv_cache, gpt2_batch["target_ids"],
        attention_mask=ctx.attention_mask, q_len=ctx.q_len,
    )
    torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        gpt2_batch["labels"].reshape(-1),
        ignore_index=-100,
    ).backward()

    gpt2_backbone.verify_frozen()
    assert_weights_unchanged(before, gpt2_backbone.weight_hash())


def test_session_not_in_module_tree(gpt2_backbone):
    """백본은 nn.Module이 아니다 — 체크포인트 오염을 구조적으로 막는다."""
    assert not isinstance(gpt2_backbone, nn.Module)
