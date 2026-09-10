"""M1 완료 조건: 빈(더미) 구현체로 조립 루트가 끝까지 통과한다.

실제 슬롯 구현은 M2 이후이므로 더미로 확인한다 — 계약 층이 하위 패키지의
구현 일정에 묶이지 않게 하기 위해서다 (ROADMAP.md M1).
"""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from lsrr.builder import build_slots, resolve_widths
from lsrr.core.errors import AssemblyError
from lsrr.core.types import ReasoningTrace


def test_builder_instantiates_every_slot(dummy_cfg):
    bundle = build_slots(dummy_cfg)
    for slot in ("encoder", "composer", "scope", "adapter", "engine", "schedule",
                 "termination", "fusion", "readout"):
        assert getattr(bundle, slot) is not None, f"슬롯 {slot}이 조립되지 않았다"


def test_widths_resolved_in_one_place(dummy_cfg):
    """d_model을 슬롯마다 따로 해석하면 조용히 어긋난다 (LEGACY_MAP §3)."""
    widths = resolve_widths(dummy_cfg, d_in=32, num_layers=6)
    assert widths == {"d_in": 32, "d_model": 32, "num_layers": 6}


def test_d_model_independent_of_d_in(dummy_cfg):
    """ADR-003: 융합 앵커가 백본 원본으로 옮겨가 d_model 제약이 풀렸다."""
    cfg = OmegaConf.merge(dummy_cfg, OmegaConf.create({"memory": {"adapter": {"d_model": 16}}}))
    bundle = build_slots(cfg)
    assert bundle.widths["d_model"] == 16 and bundle.widths["d_in"] == 32


def test_adapter_width_mismatch_fails_at_assembly(dummy_cfg):
    """폭 불일치는 학습이 아니라 조립에서 터진다."""
    from lsrr.core.interfaces import BaseLayerAdapter
    from lsrr.core.registry import ADAPTER_REGISTRY

    @ADAPTER_REGISTRY.register("_wrong_width")
    class WrongWidth(BaseLayerAdapter):
        def __init__(self, **_):
            super().__init__()

        def forward(self, H):
            return H

        @property
        def d_model(self):
            return 999

    cfg = OmegaConf.merge(dummy_cfg, OmegaConf.create({"memory": {"adapter": {"type": "_wrong_width"}}}))
    with pytest.raises(AssemblyError, match="d_model"):
        build_slots(cfg)


def test_scope_narrows_layer_axis(dummy_cfg):
    """Ablation A: final_only는 레이어 축을 1로 줄인다."""
    cfg = OmegaConf.merge(dummy_cfg, OmegaConf.create({"memory": {"scope": {"type": "_dummy_scope_final"}}}))
    bundle = build_slots(cfg)
    assert bundle.widths["scoped_layers"] == 1


def test_forward_produces_trace(dummy_model, dummy_batch):
    trace = dummy_model(dummy_batch)
    assert isinstance(trace, ReasoningTrace)
    assert trace.logits is not None and trace.h_fusion is not None
    assert trace.num_cycles > 0
    assert trace.R0.shape[:2] == (3, 6)


def test_eval_forward_terminates(dummy_model, dummy_batch):
    trace = dummy_model(dummy_batch, is_eval=True)
    assert 0 < trace.num_cycles <= dummy_model.termination.m_max


def test_trace_carries_no_module_objects(dummy_model, dummy_batch):
    """ADR-005: 트레이스에 모듈 객체를 싣지 않는다."""
    import torch.nn as nn

    trace = dummy_model(dummy_batch)
    for name, value in vars(trace).items():
        assert not isinstance(value, nn.Module), f"trace.{name}에 모듈이 실렸다"


def test_backbone_excluded_from_trainable_parameters(dummy_model):
    """백본은 nn.Module 자식이 아니다 — 체크포인트 오염 방지 (I1)."""
    trainable = dummy_model.trainable_parameters()
    assert trainable, "학습 파라미터가 하나도 없다"
    encoder_params = {id(p) for p in dummy_model.encoder.embed.parameters()}
    assert not any(id(p) in encoder_params for p in trainable)


def test_parameter_report_gives_backbone_ratio(dummy_model):
    """제안서 §5의 '백본 대비 3% 이내'가 매 런에서 확인되어야 한다."""
    report = dummy_model.parameter_report()
    assert report["trainable"] > 0 and report["backbone"] > 0
    assert report["ratio"] is not None


def test_builder_without_encoder(dummy_cfg):
    """엔진 전용 실험·프로파일링 경로."""
    bundle = build_slots(dummy_cfg, d_in=32, num_layers=6, build_encoder=False)
    assert bundle.encoder is None and bundle.engine is not None
