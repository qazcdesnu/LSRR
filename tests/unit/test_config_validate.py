"""설정 검증 계약.

학습 3시간 뒤가 아니라 로드 시점에 터져야 한다 (CONVENTIONS.md §2).
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from lsrr.config.validate import validate_config
from lsrr.core.errors import ConfigError


def _cfg(**overrides):
    base = {
        "backbone": {"type": "dummy", "d_in": 32},
        "memory": {
            "composer": {"type": "gate"},
            "scope": {"type": "all_layers"},
            "adapter": {"type": "per_layer_affine"},
        },
        "engine": {"type": "hydra_qs"},
        "recurrence": {
            "schedule": {"type": "lognormal", "max": 8},
            "tbptt_k": 2,
            "hooks": {"readout_per_cycle": False},
        },
        "termination": {"type": "delta_state", "m_max": 8},
        "readout": {
            "fusion": {"type": "attention_pooling", "fusion_type": "residual"},
            "path": {"type": "backbone_continuation"},
        },
        "data": {"type": "multiplication"},
    }
    cfg = OmegaConf.create(base)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    return cfg


def test_valid_config_passes():
    assert validate_config(_cfg()) == []


def test_unknown_top_level_key_rejected():
    with pytest.raises(ConfigError, match="알 수 없는 최상위"):
        validate_config(_cfg(enginee={"type": "typo"}))


def test_missing_required_slot_rejected():
    cfg = _cfg()
    del cfg["engine"]
    with pytest.raises(ConfigError, match="필수 슬롯"):
        validate_config(cfg)


def test_slot_without_type_rejected():
    cfg = _cfg()
    del cfg.engine["type"]
    with pytest.raises(ConfigError, match="'type' 키가 없다"):
        validate_config(cfg)


def test_m_max_below_schedule_max_rejected():
    """I5 관련: 평가 폴백이 학습 깊이보다 얕으면 분포 밖으로 나간다."""
    with pytest.raises(ConfigError, match="m_max"):
        validate_config(_cfg(termination={"m_max": 4}, recurrence={"schedule": {"max": 16}}))


def test_tbptt_window_larger_than_max_rejected():
    with pytest.raises(ConfigError, match="tbptt_k"):
        validate_config(_cfg(recurrence={"tbptt_k": 32}))


def test_residual_fusion_width_mismatch_rejected():
    """I8 / ADR-003: h_fusion은 백본 입력 임베딩 공간에 있어야 한다."""
    with pytest.raises(ConfigError, match="residual 융합"):
        validate_config(_cfg(readout={"fusion": {"d_out": 999}}))


def test_deep_supervision_requires_per_cycle_readout():
    """ADR-006: 답-앵커형 깊은 감독은 중간 사이클 판독을 요구한다."""
    with pytest.raises(ConfigError, match="readout_per_cycle"):
        validate_config(_cfg(objective={"deep_supervision": {"enabled": True}}))


def test_deep_supervision_cycles_within_tbptt_window():
    """윈도 밖 사이클은 detach되어 감독 효과가 없다."""
    with pytest.raises(ConfigError, match="num_cycles"):
        validate_config(
            _cfg(
                recurrence={"hooks": {"readout_per_cycle": True}},
                objective={"deep_supervision": {"enabled": True, "num_cycles": 9}},
            )
        )


def test_gamma_out_of_range_rejected():
    with pytest.raises(ConfigError, match="gamma"):
        validate_config(
            _cfg(
                recurrence={"hooks": {"readout_per_cycle": True}},
                objective={"deep_supervision": {"enabled": True, "gamma": 1.5}},
            )
        )


def test_output_space_termination_requires_per_cycle_readout():
    with pytest.raises(ConfigError, match="readout_per_cycle"):
        validate_config(_cfg(termination={"type": "kl_output"}))


def test_budget_match_against_self_warns_not_fails():
    """ADR-009: 기준 엔진 자신도 스윕에 포함되므로 self-match는 무연산이다."""
    warnings = validate_config(_cfg(engine={"budget_match": "hydra_qs"}))
    assert any("무연산" in w for w in warnings)


def test_budget_match_against_other_is_silent():
    assert validate_config(_cfg(engine={"budget_match": "attn_block"})) == []


def test_final_only_scope_warns_not_fails():
    """Ablation A의 의도된 축퇴는 경고이지 실패가 아니다."""
    warnings = validate_config(_cfg(memory={"scope": {"type": "final_only"}}))
    assert any("final_only" in w for w in warnings)


def test_reencoding_loop_flagged():
    """ADR-010: I2 예외는 눈에 띄어야 한다."""
    warnings = validate_config(_cfg(experimental={"reencoding_loop": True}))
    assert any("I2" in w for w in warnings)


def test_stability_ladder_flagged():
    """ADR-007: 안정화를 켠 런은 끈 런과 같은 셀에서 비교하지 않는다."""
    warnings = validate_config(_cfg(stability={"type": "jacobian"}))
    assert any("stability" in w for w in warnings)


def test_shipped_configs_validate():
    """저장소에 든 설정은 전부 검증을 통과해야 한다."""
    from lsrr.config.loader import DEFAULT_CONFIG_DIR, load_config
    from lsrr.config.sweep import expand_sweep

    for path in sorted((DEFAULT_CONFIG_DIR / "exp").glob("*.yaml")):
        validate_config(load_config([f"exp={path.stem}"], validate=False))

    for path in sorted((DEFAULT_CONFIG_DIR / "ablation").glob("*.yaml")):
        cfg = load_config([f"exp={path.stem}"], validate=False)
        for _, child in expand_sweep(cfg, path.stem):
            validate_config(child)
