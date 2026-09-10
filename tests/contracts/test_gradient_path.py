"""I4 — 역전파는 h_fusion 주입 위치를 통해서만 흐른다.

제안서 §4.4: "역전파는 h_fusion 위치를 통해서만 엔진으로 흐른다(백본 가중치 동결)."
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from lsrr.core.errors import GradientPathViolation
from lsrr.core.invariants import assert_no_grad

pytestmark = pytest.mark.contract


def _nll(logits, labels):
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100
    )


def test_backbone_accumulates_no_grad(gpt2_backbone, gpt2_batch, calibrator):
    gpt2_backbone.model.zero_grad(set_to_none=True)
    ctx = gpt2_backbone.encode(gpt2_batch["input_ids"], gpt2_batch["attention_mask"])
    h = calibrator(ctx.h_ctx.clone().requires_grad_(True))
    logits = gpt2_backbone.answer_head.teacher_forced(
        h, ctx.kv_cache, gpt2_batch["target_ids"],
        attention_mask=ctx.attention_mask, q_len=ctx.q_len,
    )
    _nll(logits, gpt2_batch["labels"]).backward()
    assert_no_grad(gpt2_backbone.model, what="gpt2")


def test_grad_reaches_injection_vector(gpt2_backbone, gpt2_batch, calibrator):
    """주입 위치로 실제 학습 신호가 도달한다 — 이것이 없으면 아래 전부가 무의미하다."""
    ctx = gpt2_backbone.encode(gpt2_batch["input_ids"], gpt2_batch["attention_mask"])
    h = ctx.h_ctx.clone().requires_grad_(True)
    logits = gpt2_backbone.answer_head.teacher_forced(
        calibrator(h), ctx.kv_cache, gpt2_batch["target_ids"],
        attention_mask=ctx.attention_mask, q_len=ctx.q_len,
    )
    _nll(logits, gpt2_batch["labels"]).backward()
    assert h.grad is not None and float(h.grad.norm()) > 0


def test_calibration_brings_injection_into_distribution(gpt2_backbone, gpt2_batch, calibrator):
    """ADR-013: 보정의 이득은 그래디언트 크기가 아니라 분포 정합이다.

    보정 전 h_ctx는 입력 임베딩 대비 노름이 약 79배로 분포 밖이며, 그 상태의
    주입은 백본을 퇴화 영역으로 밀어 넣는다.
    """
    ctx = gpt2_backbone.encode(gpt2_batch["input_ids"], gpt2_batch["attention_mask"])
    report = calibrator.scale_report(ctx.h_ctx)
    embed_norm = float(
        gpt2_backbone.model.get_input_embeddings()(gpt2_batch["input_ids"]).norm(dim=-1).mean()
    )
    assert report["ratio"] > 10, "보정 전 불일치가 크지 않다면 측정이 잘못됐다"
    assert abs(report["norm_after"] - embed_norm) / embed_norm < 0.5


def test_calibration_lowers_nll(gpt2_backbone, gpt2_batch, calibrator):
    """분포 안으로 들어간 주입이 실제로 더 나은 답 우도를 준다."""
    def nll(calibrate: bool) -> float:
        ctx = gpt2_backbone.encode(gpt2_batch["input_ids"], gpt2_batch["attention_mask"])
        h = calibrator(ctx.h_ctx) if calibrate else ctx.h_ctx
        with torch.no_grad():
            logits = gpt2_backbone.answer_head.teacher_forced(
                h, ctx.kv_cache, gpt2_batch["target_ids"],
                attention_mask=ctx.attention_mask, q_len=ctx.q_len,
            )
        return float(_nll(logits, gpt2_batch["labels"]))

    assert nll(True) < nll(False)


def test_rms_calibration_scales_upstream_gradient_down(gpt2_backbone, gpt2_batch, calibrator):
    """RMS 정규화의 야코비안은 상류 그래디언트를 gain/rms 배로 줄인다.

    이는 보정의 **부작용**이지 이득이 아니다. M3에서 학습률·클리핑을 정할 때
    알고 있어야 하는 사실이므로 회귀로 고정한다.
    """
    def upstream_grad(calibrate: bool) -> float:
        ctx = gpt2_backbone.encode(gpt2_batch["input_ids"], gpt2_batch["attention_mask"])
        h = ctx.h_ctx.clone().requires_grad_(True)
        logits = gpt2_backbone.answer_head.teacher_forced(
            calibrator(h) if calibrate else h, ctx.kv_cache, gpt2_batch["target_ids"],
            attention_mask=ctx.attention_mask, q_len=ctx.q_len,
        )
        _nll(logits, gpt2_batch["labels"]).backward()
        return float(h.grad.norm())

    assert upstream_grad(True) < upstream_grad(False)


def test_fixed_scale_preserves_gradient_direction(gpt2_backbone, gpt2_batch):
    """fixed_scale의 야코비안은 c·I이므로 상류 그래디언트 방향이 보존된다."""
    from lsrr.readout import InjectionCalibrator

    cal = InjectionCalibrator.from_backbone(gpt2_backbone.model, mode="fixed_scale")
    ctx = gpt2_backbone.encode(gpt2_batch["input_ids"], gpt2_batch["attention_mask"])
    cal.calibrate_from_batch(ctx.h_ctx)

    grads = []
    for calibrate in (True, False):
        c = gpt2_backbone.encode(gpt2_batch["input_ids"], gpt2_batch["attention_mask"])
        h = c.h_ctx.clone().requires_grad_(True)
        logits = gpt2_backbone.answer_head.teacher_forced(
            cal(h) if calibrate else h, c.kv_cache, gpt2_batch["target_ids"],
            attention_mask=c.attention_mask, q_len=c.q_len,
        )
        _nll(logits, gpt2_batch["labels"]).backward()
        grads.append(h.grad.flatten())

    # 보정 여부와 무관하게 같은 지점에서 잰 것이 아니므로 방향만 대략 비교한다
    cos = torch.nn.functional.cosine_similarity(grads[0], grads[1], dim=0)
    assert float(cos) > 0.0


def test_assert_no_grad_catches_violation():
    import torch.nn as nn

    m = nn.Linear(4, 4)
    m(torch.randn(2, 4)).sum().backward()
    with pytest.raises(GradientPathViolation, match="h_fusion"):
        assert_no_grad(m)
