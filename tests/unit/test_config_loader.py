"""설정 로드·병합 계약."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from lsrr.config.loader import (
    DEFAULT_CONFIG_DIR,
    load_config,
    parse_cli,
    resolve_hierarchical,
)
from lsrr.core.errors import ConfigError


def test_parse_cli_splits_target_and_overrides():
    target, overrides = parse_cli(["exp=smoke", "engine.type=mamba_up", "train.lr=5e-4"])
    assert target == "smoke"
    assert overrides == ["engine.type=mamba_up", "train.lr=5e-4"]


def test_parse_cli_rejects_bare_argument():
    """오타를 조용히 삼키지 않는다."""
    with pytest.raises(ConfigError, match="해석할 수 없다"):
        parse_cli(["--verbose"])


def test_base_config_loads():
    cfg = load_config([], default_exp="base", validate=False)
    assert cfg.seed == 42
    assert cfg.termination.m_max == 32


def test_defaults_merge_in_order(tmp_path: Path):
    (tmp_path / "a.yaml").write_text("x: 1\ny: 1\n")
    (tmp_path / "b.yaml").write_text("defaults: [a]\ny: 2\n")
    cfg = resolve_hierarchical(tmp_path / "b.yaml", tmp_path)
    assert (cfg.x, cfg.y) == (1, 2)  # 나중 파일이 이긴다


def test_cli_override_wins_over_files():
    cfg = load_config(["exp=smoke", "engine.type=mamba_up"], validate=False)
    assert cfg.engine.type == "mamba_up"


def test_smoke_exp_expands_backbone_section():
    """defaults의 backbone/gpt2가 실제로 병합된다."""
    cfg = load_config(["exp=smoke"], validate=False)
    assert cfg.backbone.model_name_or_path == "gpt2"
    assert cfg.backbone.d_in == 768


def test_circular_defaults_detected(tmp_path: Path):
    (tmp_path / "a.yaml").write_text("defaults: [b]\n")
    (tmp_path / "b.yaml").write_text("defaults: [a]\n")
    with pytest.raises(ConfigError, match="순환"):
        resolve_hierarchical(tmp_path / "a.yaml", tmp_path)


def test_missing_config_file_names_what_it_looked_for(tmp_path: Path):
    with pytest.raises(ConfigError, match="찾지 못했다"):
        resolve_hierarchical(tmp_path / "nope.yaml", tmp_path)


def test_string_section_expands_to_file():
    """`engine: hydra_qs` 축약이 configs/engine/hydra_qs.yaml로 확장된다."""
    engine_dir = DEFAULT_CONFIG_DIR / "engine"
    engine_dir.mkdir(parents=True, exist_ok=True)
    path = engine_dir / "_test_tmp.yaml"
    path.write_text("engine:\n  type: from_file\n  d_state: 99\n")
    try:
        cfg = load_config(["exp=smoke", "engine=_test_tmp"], validate=False)
        assert cfg.engine.type == "from_file"
        assert cfg.engine.d_state == 99
    finally:
        path.unlink()
