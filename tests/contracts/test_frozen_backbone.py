"""I1 — 백본은 동결이며 학습 전후 가중치가 변하지 않는다.

제안서 §5: 백본은 완전 동결하며(LoRA 불포함), 학습 대상은 레이어 어댑터 +
SSM 엔진 + 풀링/융합 헤드뿐이다(백본 대비 약 3% 이내).
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
    with pytest.raises(FrozenBackboneViolation, match="LoRA 불포함"):
        assert_frozen(nn.Linear(4, 4))


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
