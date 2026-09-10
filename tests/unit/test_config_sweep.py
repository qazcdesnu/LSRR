"""스윕 전개 계약 — 실험은 설정 조합이다 (ADR-009)."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from lsrr.config.sweep import expand_sweep, sweep_size
from lsrr.core.errors import ConfigError


def test_no_sweep_returns_single():
    cfg = OmegaConf.create({"engine": {"type": "hydra_qs"}})
    result = expand_sweep(cfg, "exp")
    assert len(result) == 1 and result[0][0] == "exp"


def test_cartesian_product():
    cfg = OmegaConf.create(
        {
            "engine": {"type": "hydra_qs"},
            "sweep": {"engine.type": ["a", "b"], "engine.damping_alpha": [0.3, 0.7]},
        }
    )
    result = expand_sweep(cfg, "abl")
    assert len(result) == 4 == sweep_size(cfg)
    assert {c.engine.type for _, c in result} == {"a", "b"}
    assert all("sweep" not in c for _, c in result)


def test_child_names_encode_the_axis():
    cfg = OmegaConf.create({"sweep": {"engine.type": ["mamba_up"]}})
    (name, _), = expand_sweep(cfg, "C")
    assert name == "C_type_mamba_up"


def test_empty_axis_rejected():
    cfg = OmegaConf.create({"sweep": {"engine.type": []}})
    with pytest.raises(ConfigError, match="비어 있다"):
        expand_sweep(cfg)


def test_ablation_c_expands_to_six_engines():
    """Ablation C는 믹서 행렬 클래스 6종을 비교한다 (제안서 §7)."""
    from lsrr.config.loader import load_config

    cfg = load_config(["exp=C_engine"], validate=False)
    assert sweep_size(cfg) == 6
