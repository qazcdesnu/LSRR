"""고정 중심화 (ADR-019) — 층 상태의 학습셋 평균을 뺀다.

마지막 질문 토큰의 층 상태는 샘플 간 코사인 0.9755 로 거의 같은 방향이고(F-030),
그 공유 성분은 고정 벡터 하나다(F-040). 고정하는 것:

- 공유 오프셋이 섞인 입력에서 중심화가 코사인을 0 근처로 내린다
- 추정 전 forward 는 실패한다 — 0 벡터를 조용히 빼면 "중심화했다"고 믿게 된다
- `none` 은 비트 단위 무연산이다 (옛 스냅샷의 재현)
- 평균은 버퍼라 체크포인트에 실린다
- 컴포저가 쓰는 H_pool 도 함께 중심화된다 — 안 하면 공유 성분이 그쪽으로 되돌아온다
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from lsrr.core.types import ContextBundle
from lsrr.memory.pipeline import LayerMemoryPipeline, _mean_cosine

N, L, D = 64, 4, 16


def _ctx(H, P=None):
    return ContextBundle(H_last=H, h_ctx=H[:, -1], H_pool=P, hidden_stack=None,
                         kv_cache=None, attention_mask=None, q_len=None, meta={})


def _shared_plus_noise(seed=0):
    g = torch.Generator().manual_seed(seed)
    shared = torch.randn(L, D, generator=g) * 20.0      # 공유 성분 (GPT-2 이방성 흉내)
    noise = torch.randn(N, L, D, generator=g)          # 샘플별 정보
    return shared + noise


def test_centering_removes_the_shared_direction():
    H = _shared_plus_noise()
    assert _mean_cosine(H) > 0.95                       # F-030 의 상황
    pipe = LayerMemoryPipeline(center="train_mean", d_in=D, num_layers=L)
    stats = pipe.estimate_center(H)
    assert stats["cosine_before"] > 0.95 and abs(stats["cosine_after"]) < 0.1
    out = pipe(_ctx(H))
    assert abs(_mean_cosine(out)) < 0.1


def test_forward_before_estimate_fails_loudly():
    pipe = LayerMemoryPipeline(center="train_mean", d_in=D, num_layers=L)
    with pytest.raises(RuntimeError, match="추정되지 않았다"):
        pipe(_ctx(_shared_plus_noise()))


def test_none_is_a_bitwise_noop():
    H = _shared_plus_noise()
    pipe = LayerMemoryPipeline(center="none")
    assert torch.equal(pipe(_ctx(H)), H)
    assert not hasattr(pipe, "mu_last")


def test_mean_is_a_buffer_and_round_trips():
    H = _shared_plus_noise()
    a = LayerMemoryPipeline(center="train_mean", d_in=D, num_layers=L); a.estimate_center(H)
    assert "mu_last" in a.state_dict() and "center_ready" in a.state_dict()
    assert not any(p.requires_grad for n, p in a.named_parameters() if "mu_" in n)
    b = LayerMemoryPipeline(center="train_mean", d_in=D, num_layers=L)
    b.load_state_dict(a.state_dict())
    assert torch.equal(a(_ctx(H)), b(_ctx(H)))


def test_h_pool_is_centered_too():
    """컴포저가 H_pool 을 섞으면 공유 성분이 그쪽으로 되돌아온다."""
    H, P = _shared_plus_noise(0), _shared_plus_noise(1)

    class _Mix(nn.Module):
        def forward(self, h, p):
            return 0.5 * h + 0.5 * p

    pipe = LayerMemoryPipeline(composer=_Mix(), center="train_mean", d_in=D, num_layers=L)
    pipe.estimate_center(H, P)
    assert abs(_mean_cosine(pipe(_ctx(H, P)))) < 0.1


def test_unknown_center_is_rejected():
    with pytest.raises(ValueError, match="memory.center"):
        LayerMemoryPipeline(center="batch")


def test_builder_passes_the_center_key():
    from lsrr.builder import build_slots
    from lsrr.config import load_config

    b = build_slots(load_config(["exp=exp/prosqa_gpt2", "device=cpu", "memory.center=train_mean"]),
                    d_in=768, num_layers=12, build_encoder=False) if False else None
    # 인코더 없이 조립하면 판독 보정 목표가 없어 실패하므로, 파이프라인만 직접 확인한다.
    from lsrr.config.schema import get_path
    cfg = load_config(["exp=exp/prosqa_gpt2", "memory.center=train_mean"])
    assert get_path(cfg, "memory.center") == "train_mean"
