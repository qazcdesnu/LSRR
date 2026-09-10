"""스냅샷과 런 신원."""

from __future__ import annotations

from omegaconf import OmegaConf

from lsrr.config.snapshot import (
    config_hash,
    load_snapshot,
    make_run_id,
    run_metadata,
    save_snapshot,
)


def test_hash_is_order_independent():
    a = OmegaConf.create({"x": 1, "y": {"b": 2, "a": 1}})
    b = OmegaConf.create({"y": {"a": 1, "b": 2}, "x": 1})
    assert config_hash(a) == config_hash(b)


def test_hash_changes_with_content():
    a = OmegaConf.create({"engine": {"type": "hydra_qs"}})
    b = OmegaConf.create({"engine": {"type": "mamba_up"}})
    assert config_hash(a) != config_hash(b)


def test_run_id_format():
    cfg = OmegaConf.create({"engine": {"type": "hydra_qs"}, "backbone": {"type": "gpt2"}})
    rid = make_run_id(cfg, "prosqa", seed=3)
    assert rid.startswith("prosqa_hydra_qs_gpt2_") and rid.endswith("_s3")


def test_snapshot_roundtrip(tmp_path):
    cfg = OmegaConf.create({"engine": {"type": "hydra_qs"}, "train": {"lr": 3e-4}})
    path = save_snapshot(cfg, tmp_path / "run" / "config.yaml")
    assert config_hash(load_snapshot(path)) == config_hash(cfg)


def test_metadata_records_comparability_flags():
    """표에 병기해야 하는 값들이 메타에 고정된다 (ADR-007, ADR-010)."""
    cfg = OmegaConf.create(
        {
            "engine": {"type": "hydra_qs"},
            "backbone": {"type": "gpt2"},
            "stability": {"type": "jacobian"},
            "experimental": {"reencoding_loop": True},
        }
    )
    meta = run_metadata(cfg, "exp", seed=0)
    assert meta["stability_rung"] == "jacobian"
    assert meta["reencoding_loop"] is True
    assert len(meta["config_hash"]) == 12
