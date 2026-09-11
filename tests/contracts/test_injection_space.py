"""I8 — h_fusion은 백본 입력 임베딩 공간의 벡터다.

ADR-003: 융합 잔차 앵커는 어댑터 통과 **전**의 백본 원본 h^(L)이며,
W_r의 출력 폭은 d_in이다. 어댑터 공간 벡터를 주입하면 매니폴드 불일치가
발생한다 — 제안서 §9가 "원천 차단"을 기여로 내건 실패 양식이다.
"""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from lsrr.core.errors import InjectionSpaceError
from lsrr.core.invariants import assert_injection_space

pytestmark = pytest.mark.contract


def test_correct_width_passes():
    assert_injection_space(torch.zeros(4, 768), d_in=768)


def test_adapter_space_width_rejected():
    with pytest.raises(InjectionSpaceError, match="ADR-003"):
        assert_injection_space(torch.zeros(4, 256), d_in=768)


def test_trajectory_of_correct_width_passes():
    """v2.1 §4.4: M개 잠재 사고 토큰 — 모든 토큰이 백본 공간에 있어야 한다 (ADR-015)."""
    for M in (1, 3, 8):
        assert_injection_space(torch.zeros(4, M, 768), d_in=768)


def test_trajectory_of_wrong_width_rejected():
    with pytest.raises(InjectionSpaceError, match="ADR-003"):
        assert_injection_space(torch.zeros(4, 3, 256), d_in=768)


def test_wrong_rank_rejected():
    """[B, d] 와 [B, M, d] 만 허용한다."""
    with pytest.raises(InjectionSpaceError, match="d_in"):
        assert_injection_space(torch.zeros(768), d_in=768)
    with pytest.raises(InjectionSpaceError, match="d_in"):
        assert_injection_space(torch.zeros(2, 4, 12, 768), d_in=768)


def test_empty_trajectory_rejected():
    """M_min ≥ 1 — 아무것도 방출하지 않으면 백본이 읽을 것이 없다."""
    with pytest.raises(InjectionSpaceError, match="비어 있다"):
        assert_injection_space(torch.zeros(4, 0, 768), d_in=768)


def test_model_readout_lands_in_backbone_space(dummy_model, dummy_batch):
    trace = dummy_model(dummy_batch)
    assert trace.h_fusion.shape[-1] == dummy_model.d_in


def test_narrow_engine_still_injects_at_backbone_width(dummy_cfg, dummy_batch, runner_factory):
    """d_model < d_in이어도 h_fusion은 d_in 폭이어야 한다."""
    from lsrr.builder import build_slots
    from lsrr.model import LSRRModel

    cfg = OmegaConf.merge(dummy_cfg, OmegaConf.create({"memory": {"adapter": {"d_model": 8}}}))
    bundle = build_slots(cfg)
    model = LSRRModel(bundle=bundle, cfg=cfg, runner=runner_factory(bundle))
    trace = model(dummy_batch)
    assert model.d_model == 8
    assert trace.h_fusion.shape[-1] == model.d_in == 32


def test_fusion_residual_starts_near_backbone_behaviour(dummy_model, dummy_batch):
    """W_r을 작게 초기화하므로 학습 초기 h_fusion ≈ h_ctx (readout/README)."""
    context = dummy_model.encode(dummy_batch["input_ids"])
    R0 = dummy_model.build_memory(context)
    result = dummy_model.read(R0, context)
    drift = (result.h_fusion - context.h_ctx).norm() / context.h_ctx.norm().clamp(min=1e-6)
    assert drift < 0.5
