"""2원화 학습의 페이즈 (v2.1 §5.0, ADR-014).

이 분해가 **Ablation B 를 학습 과정 자체에서 산출한다** — Phase A 체크포인트가
완전 동결 조건이고 Phase B 증분이 순수 정렬 이득이다. 그러려면 각 페이즈가
무엇을 학습하는지가 한 곳에 고정돼 있어야 한다.
"""

from __future__ import annotations

import pytest
import torch.nn as nn
from omegaconf import OmegaConf

from lsrr.core.errors import ConfigError
from lsrr.runtime.phases import (
    GROUPS,
    PhaseSpec,
    apply_phase,
    phases_from_cfg,
)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.engine = nn.Linear(4, 4)
        self.pipeline = nn.Linear(4, 4)
        self.pooler = nn.Linear(4, 4)
        self.fusion = nn.Linear(4, 4)
        self.readout = nn.Linear(4, 4)


class _Encoder:
    def __init__(self, n: int = 3):
        self.has_lora = True
        self._params = [nn.Parameter(nn.init.zeros_(nn.Parameter(nn.Linear(2, 2).weight)))
                        for _ in range(n)]

    def lora_parameters(self):
        return self._params


def _trainable(model):
    return {n.split(".")[0] for n, p in model.named_parameters() if p.requires_grad}


def test_phase_a_trains_the_engine_stack():
    model = _Model()
    apply_phase(model, None, PhaseSpec("A", 1, 1e-3, tuple(GROUPS)))
    assert _trainable(model) == {"engine", "pipeline", "pooler", "fusion", "readout"}


def test_phase_b_freezes_the_engine():
    """Phase B 증분이 '순수 정렬 이득' 이려면 엔진이 얼어 있어야 한다."""
    model = _Model()
    apply_phase(model, _Encoder(), PhaseSpec("B", 1, 1e-4, ("emitter", "lora"), True))
    assert _trainable(model) == {"fusion", "readout"}


def test_previous_phase_state_is_not_inherited():
    """먼저 끄고 켜지 않으면 페이즈 순서에 따라 학습 대상이 달라진다."""
    model = _Model()
    apply_phase(model, None, PhaseSpec("A", 1, 1e-3, tuple(GROUPS)))
    apply_phase(model, _Encoder(), PhaseSpec("B", 1, 1e-4, ("emitter", "lora"), True))
    assert not any(p.requires_grad for p in model.engine.parameters())


def test_lora_params_reach_the_optimizer_list():
    """백본은 nn.Module 이 아니라 `model.parameters()` 에 잡히지 않는다 (I1).

    페이즈가 명시적으로 넘기지 않으면 Phase B 학습이 통째로 버려진다.
    """
    enc = _Encoder(n=3)
    out = apply_phase(_Model(), enc, PhaseSpec("B", 1, 1e-4, ("emitter", "lora"), True))
    assert out["by_group"]["lora"] == sum(p.numel() for p in enc.lora_parameters())
    assert all(any(p is q for q in out["params"]) for p in enc.lora_parameters())


def test_lora_is_frozen_when_not_in_trainable():
    """Phase C 대조처럼 어댑터를 붙여만 두고 얼리는 구성도 있어야 한다."""
    enc = _Encoder()
    apply_phase(_Model(), enc, PhaseSpec("x", 1, 1e-4, ("emitter",), attach_lora=True))
    assert not any(p.requires_grad for p in enc.lora_parameters())


def test_training_lora_without_attaching_is_rejected():
    with pytest.raises(ConfigError, match="붙이지 않은 어댑터"):
        PhaseSpec("B", 1, 1e-4, ("emitter", "lora"), attach_lora=False)


def test_unknown_group_is_rejected():
    with pytest.raises(ConfigError, match="trainable 그룹"):
        PhaseSpec("A", 1, 1e-3, ("backbone",))


def test_empty_phase_is_rejected():
    with pytest.raises(ConfigError, match="학습할 파라미터가 하나도 없다"):
        apply_phase(_Model(), None, PhaseSpec("z", 1, 1e-3, ()))


def test_missing_phases_section_falls_back_to_one_phase():
    """기존 1-페이즈 설정이 그대로 돌아야 한다."""
    cfg = OmegaConf.create({"train": {"epochs": 7, "lr": 1e-3}})
    (spec,) = phases_from_cfg(cfg)
    assert spec.epochs == 7 and spec.lr == pytest.approx(1e-3)
    assert set(spec.trainable) == set(GROUPS) and not spec.attach_lora


def test_phases_section_is_read_in_order():
    cfg = OmegaConf.create({
        "train": {"epochs": 1, "lr": 1e-3, "phases": [
            {"name": "A", "epochs": 10, "lr": 1e-3,
             "trainable": ["engine", "memory", "emitter"]},
            {"name": "B", "epochs": 5, "lr": 1e-4, "trainable": ["emitter", "lora"]},
        ]},
    })
    a, b = phases_from_cfg(cfg)
    assert (a.name, a.epochs, a.attach_lora) == ("A", 10, False)
    assert (b.name, b.epochs, b.attach_lora) == ("B", 5, True)


def test_overlapping_attributes_are_not_double_counted():
    """`pipeline` 이 `composer`·`adapter` 를 자식으로 품고 셋이 모두 속성이다.

    중복이 남으면 옵티마이저가 같은 파라미터를 두 번 갱신해 그 모듈만 유효
    학습률이 2배가 된다 — 조용히, 그리고 모듈마다 다르게.
    """

    class _Nested(nn.Module):
        def __init__(self):
            super().__init__()
            self.adapter = nn.Linear(4, 4)
            self.pipeline = nn.Sequential(self.adapter)
            self.pooler = nn.Linear(4, 4)
            self.engine = nn.Linear(4, 4)
            self.fusion = nn.Linear(4, 4)
            self.readout = nn.Linear(4, 4)

    model = _Nested()
    out = apply_phase(model, None, PhaseSpec("A", 1, 1e-3, ("memory",)))
    assert len({id(p) for p in out["params"]}) == len(out["params"])
    expected = sum(p.numel() for p in {id(p): p for p in
                   list(model.pipeline.parameters()) + list(model.pooler.parameters())}.values())
    assert out["by_group"]["memory"] == expected
