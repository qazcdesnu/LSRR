"""백본 세션 — 1회 인코딩과 연속 디코딩의 수치 정합."""

from __future__ import annotations

import pytest
import torch

from lsrr.backbone.extractor import extract_last_token, stack_hidden_states
from lsrr.core.errors import LSRRError


def test_context_bundle_shapes(gpt2_backbone, gpt2_batch):
    ctx = gpt2_backbone.encode(gpt2_batch["input_ids"], gpt2_batch["attention_mask"])
    B, T = gpt2_batch["input_ids"].shape
    L, d = gpt2_backbone.num_layers, gpt2_backbone.hidden_dim
    assert ctx.H_last.shape == (B, L, d)
    assert ctx.hidden_stack.shape == (B, L, T, d)
    assert ctx.h_ctx.shape == (B, d)
    assert ctx.meta["kv_len"] == T


def test_h_ctx_is_pre_adapter_backbone_output(gpt2_backbone, gpt2_batch):
    """ADR-003: 융합 앵커는 어댑터 통과 전 백본 원본 h(L)이다."""
    ctx = gpt2_backbone.encode(gpt2_batch["input_ids"], gpt2_batch["attention_mask"])
    assert torch.allclose(ctx.h_ctx, ctx.H_last[:, -1, :])


def test_right_padded_batch_rejected(gpt2_backbone):
    """ADR-011: 우측 패딩은 KV 이어붙이기를 깨뜨린다."""
    ids = torch.tensor([[10, 11, 12], [13, 14, 0]])
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    with pytest.raises(LSRRError, match="좌측 패딩"):
        gpt2_backbone.encode(ids, mask)


def test_kv_continuation_matches_full_forward(gpt2_backbone, gpt2_batch):
    """ADR-011 회귀: KV 재사용 경로가 전체 순전파와 일치한다.

    좌측 패딩·position_ids·어텐션 마스크가 모두 맞아야만 통과한다 — 이 셋 중
    하나라도 어긋나면 조용히 틀린 로짓으로 학습이 진행된다.
    """
    model = gpt2_backbone.model
    ids, mask = gpt2_batch["input_ids"], gpt2_batch["attention_mask"]
    nxt = torch.full((ids.shape[0], 1), 50, dtype=torch.long)

    ctx = gpt2_backbone.encode(ids, mask)
    mask2 = torch.cat([mask, torch.ones_like(nxt)], dim=1)
    with torch.no_grad():
        via_kv = model(
            input_ids=nxt,
            attention_mask=mask2,
            position_ids=ctx.q_len.unsqueeze(1),
            past_key_values=ctx.kv_cache,
            use_cache=True,
        ).logits[:, -1]

        full_pos = (mask2.cumsum(-1) - 1).clamp(min=0)
        full = model(
            input_ids=torch.cat([ids, nxt], dim=1),
            attention_mask=mask2,
            position_ids=full_pos,
        ).logits[:, -1]

    gap = float((via_kv - full).abs().max())
    assert gap < 1e-3, f"KV 경로와 전체 순전파 불일치: {gap:.2e}"


def test_cache_rewinds_for_repeated_readout(gpt2_backbone, gpt2_batch, calibrator):
    """깊은 감독은 같은 질문 KV로 여러 사이클을 판독한다 (ADR-006).

    되감지 않으면 두 번째 판독이 첫 번째의 토큰이 붙은 캐시를 본다 —
    조용히 오염된 채 학습이 진행된다.
    """
    ctx = gpt2_backbone.encode(gpt2_batch["input_ids"], gpt2_batch["attention_mask"])
    base = ctx.meta["kv_len"]
    h = calibrator(ctx.h_ctx)

    outs = []
    for _ in range(3):
        with torch.no_grad():
            outs.append(
                gpt2_backbone.answer_head.teacher_forced(
                    h,
                    ctx.kv_cache,
                    gpt2_batch["target_ids"],
                    attention_mask=ctx.attention_mask,
                    q_len=ctx.q_len,
                )
            )
        assert ctx.kv_cache.get_seq_length() == base

    assert torch.allclose(outs[0], outs[1]) and torch.allclose(outs[1], outs[2])


def test_generate_rewinds_cache(gpt2_backbone, gpt2_batch, calibrator):
    ctx = gpt2_backbone.encode(gpt2_batch["input_ids"], gpt2_batch["attention_mask"])
    base = ctx.meta["kv_len"]
    gpt2_backbone.answer_head.generate(
        calibrator(ctx.h_ctx),
        ctx.kv_cache,
        max_new_tokens=4,
        attention_mask=ctx.attention_mask,
        q_len=ctx.q_len,
    )
    assert ctx.kv_cache.get_seq_length() == base


def test_teacher_forced_logits_align_with_labels(gpt2_backbone, gpt2_batch, calibrator):
    """위치 t의 로짓이 answer_ids[:, t]를 예측한다 — 별도 shift 불필요."""
    ctx = gpt2_backbone.encode(gpt2_batch["input_ids"], gpt2_batch["attention_mask"])
    with torch.no_grad():
        logits = gpt2_backbone.answer_head.teacher_forced(
            calibrator(ctx.h_ctx),
            ctx.kv_cache,
            gpt2_batch["target_ids"],
            attention_mask=ctx.attention_mask,
            q_len=ctx.q_len,
        )
    assert logits.shape[:2] == gpt2_batch["labels"].shape


def test_include_embedding_changes_layer_count():
    hs = [torch.randn(2, 3, 8) for _ in range(5)]  # L=4 + 임베딩
    assert stack_hidden_states(hs, include_embedding=False).shape[1] == 4
    assert stack_hidden_states(hs, include_embedding=True).shape[1] == 5


def test_extractor_rejects_padded_last_position():
    stack = torch.randn(2, 3, 4, 8)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
    with pytest.raises(LSRRError, match="ADR-011"):
        extract_last_token(stack, mask)


def test_pooler_ignores_padding(gpt2_backbone, gpt2_batch):
    """ADR-004: 질문 전체 풀링은 패딩을 보지 않는다."""
    from lsrr.backbone import AttentionPooler

    ctx = gpt2_backbone.encode(gpt2_batch["input_ids"], gpt2_batch["attention_mask"])
    pooler = AttentionPooler(
        d_in=gpt2_backbone.hidden_dim, num_layers=gpt2_backbone.num_layers
    )
    pooled = pooler(ctx.hidden_stack, ctx.attention_mask)
    assert pooled.shape == ctx.H_last.shape

    alpha = pooler.last_alpha
    pad = (ctx.attention_mask == 0)[:, None, :].expand_as(alpha)
    assert float(alpha[pad].sum()) == pytest.approx(0.0, abs=1e-6)


def test_pooler_differs_from_last_token(gpt2_backbone, gpt2_batch):
    """보완항이 실제로 다른 정보를 담는다 — 같다면 결합할 이유가 없다."""
    from lsrr.backbone import AttentionPooler

    ctx = gpt2_backbone.encode(gpt2_batch["input_ids"], gpt2_batch["attention_mask"])
    pooled = AttentionPooler(
        d_in=gpt2_backbone.hidden_dim, num_layers=gpt2_backbone.num_layers
    )(ctx.hidden_stack, ctx.attention_mask)
    assert not torch.allclose(pooled, ctx.H_last, atol=1e-3)
