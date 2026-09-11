"""방출 구조 축 — Ablation A (ADR-015·ADR-016).

세 조건이 기존 두 축(`readout.path.emission` × `termination.type`)의 조합으로
표현되며 별도 코드 경로가 없다 (ADR-009).

| 조건 | emission | termination | anchor |
|---|---|---|---|
| single (v1 대조군) | `single` | — | `ctx` |
| fixed K | `trajectory` | `fixed_m` | `first` |
| dynamic M | `trajectory` | `delta_state` | `first` |
"""

from __future__ import annotations

import pytest
import torch

from lsrr.core.errors import AssemblyError, ConfigError
from lsrr.config import load_config


def _cfg(*overrides: str):
    # 테스트는 GPU 없이 CPU 에서 통과해야 한다 (규약 §4).
    return load_config(
        ["exp=ablation/A_emission", "data.sizes.train=200", "device=cpu", *overrides]
    )


# ---------------------------------------------------------------- 설정 축

def test_ablation_a_config_resolves():
    from lsrr.config.schema import get_path

    cfg = _cfg()
    assert get_path(cfg, "readout.path.emission") == "trajectory"
    assert get_path(cfg, "readout.fusion.anchor") == "first"
    assert get_path(cfg, "gates.phase0.min_seeds") == 5


def test_misplaced_emission_key_is_fatal():
    """`readout.emission` 은 빌더가 읽지 않는다 — 조용히 무시되면 조건이 바뀐다 (F-028)."""
    with pytest.raises(ConfigError, match="readout 아래 알 수 없는 키"):
        _cfg("readout.emission=single")


def test_unknown_emission_is_fatal():
    from lsrr.readout.path import BackboneContinuationReadout

    with pytest.raises(AssemblyError, match="emission"):
        BackboneContinuationReadout(
            d_in=8, calibration="none", emission="bogus"
        )


# ---------------------------------------------------------------- 방출 개수

@pytest.mark.parametrize(
    "overrides,expect_tokens",
    [
        (("readout.path.emission=single", "readout.fusion.anchor=ctx"), 1),
        (("readout.path.emission=trajectory", "termination.type=fixed_m",
          "termination.m=4"), 4),
    ],
)
def test_emission_controls_token_count(overrides, expect_tokens):
    from lsrr.builder import build_slots
    from lsrr.data import PromptEncoder, PromptSpec
    from lsrr.data.collate import make_loader
    from lsrr.model import LSRRModel

    cfg = _cfg(*overrides)
    torch.manual_seed(0)
    bundle = build_slots(cfg)
    model = LSRRModel(bundle=bundle, cfg=cfg, runner=bundle.runner)
    enc = PromptEncoder(
        bundle.encoder.tokenizer,
        PromptSpec(max_question_tokens=32, max_answer_tokens=12),
    )
    loader = make_loader(bundle.data.get_split("val")[:4], enc, batch_size=4, shuffle=False)
    with torch.no_grad():
        trace = model(next(iter(loader)), is_eval=True)
    assert trace.h_thought.shape[1] == expect_tokens


def test_dynamic_m_emits_one_token_per_cycle():
    """동적 M — 방출 개수가 종료 규칙이 정한 사이클 수와 같아야 한다."""
    from lsrr.builder import build_slots
    from lsrr.data import PromptEncoder, PromptSpec
    from lsrr.data.collate import make_loader
    from lsrr.model import LSRRModel

    cfg = _cfg()
    torch.manual_seed(0)
    bundle = build_slots(cfg)
    model = LSRRModel(bundle=bundle, cfg=cfg, runner=bundle.runner)
    enc = PromptEncoder(
        bundle.encoder.tokenizer,
        PromptSpec(max_question_tokens=32, max_answer_tokens=12),
    )
    loader = make_loader(bundle.data.get_split("val")[:4], enc, batch_size=4, shuffle=False)
    with torch.no_grad():
        trace = model(next(iter(loader)), is_eval=True)
    assert trace.h_thought.shape[1] == trace.num_cycles


def test_logits_align_with_answer_regardless_of_emission():
    """주입 토큰이 몇 개든 로짓은 답 길이와 정렬된다."""
    from lsrr.builder import build_slots
    from lsrr.data import PromptEncoder, PromptSpec
    from lsrr.data.collate import make_loader
    from lsrr.model import LSRRModel

    for overrides in (
        ("readout.path.emission=single", "readout.fusion.anchor=ctx"),
        ("readout.path.emission=trajectory", "termination.type=fixed_m",
         "termination.m=5"),
    ):
        cfg = _cfg(*overrides)
        torch.manual_seed(0)
        bundle = build_slots(cfg)
        model = LSRRModel(bundle=bundle, cfg=cfg, runner=bundle.runner)
        enc = PromptEncoder(
            bundle.encoder.tokenizer,
            PromptSpec(max_question_tokens=32, max_answer_tokens=12),
        )
        loader = make_loader(
            bundle.data.get_split("val")[:4], enc, batch_size=4, shuffle=False
        )
        batch = next(iter(loader))
        with torch.no_grad():
            trace = model(batch, is_eval=True)
        assert trace.logits.shape[1] == batch["target_ids"].shape[1], overrides


# ---------------------------------------------------------------- 킬 스위치 술어

def test_gate3_is_emission_neutral():
    """게이트 ③ 의 판정 대상이 v2 에서 교체된다 (ADR-016) — 술어는 재사용한다."""
    from lsrr.gates import gate_beats_baseline

    r = gate_beats_baseline(
        [0.31, 0.29, 0.33, 0.30, 0.32], [0.21, 0.22, 0.20, 0.215, 0.205],
        treatment_name="dynamic_m", control_name="single",
    )
    assert r.passed and r.is_kill_switch
    assert "dynamic_m" in r.detail and "single" in r.detail
    assert r.evidence["treatment"] == "dynamic_m"


def test_gate3_requires_five_seeds():
    """시드 3개로는 검정력이 부족하다 — 1자리 실측에서 d=1.31 인데 p=0.0543 이었다."""
    from lsrr.gates import Phase0Thresholds

    assert Phase0Thresholds().min_seeds == 5
