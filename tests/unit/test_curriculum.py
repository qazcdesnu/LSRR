"""단계적 잠재화 커리큘럼 (v2.1 §5.1).

고정하는 것:
- 타깃이 스테이지대로 렌더링된다 (남은 CoT + 답, S3 는 답만)
- `expand: S2` 가 k=1..K 로 펼쳐지고 M=k+1 이다
- 스테이지가 러너의 M 을 고정/복원한다 — 학습·평가가 같은 M 을 써야 한다 (F-031)
- 스테이지 루프가 Phase A 안에서만 돈다
"""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from lsrr.core.errors import ConfigError
from lsrr.core.types import DataSample
from lsrr.data.prompting import PromptSpec
from lsrr.runtime.curriculum import StageSpec, apply_stage, stages_from_cfg

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLE = DataSample(question="Q", answer="Sally is a sterpus.",
                    cot_steps=["Sally is a scrompus.", "Every scrompus is a rempus.", "Every rempus is a sterpus."])


@pytest.mark.parametrize("k,expected", [
    (None, " Sally is a sterpus."),
    (0, " Sally is a scrompus. Every scrompus is a rempus. Every rempus is a sterpus. Sally is a sterpus."),
    (1, " Every scrompus is a rempus. Every rempus is a sterpus. Sally is a sterpus."),
    (3, " Sally is a sterpus."),
    (9, " Sally is a sterpus."),   # 체인이 k 보다 짧으면 전부 제거 (Coconut 각주 1)
])
def test_target_follows_the_stage(k, expected):
    assert PromptSpec(cot_keep_from=k).render_answer(SAMPLE) == expected


def _cfg(**curr):
    return OmegaConf.create({
        "prompt": {"max_answer_tokens": 12},
        "recurrence": {"schedule": {"type": "lognormal", "mean": 5, "max": 8}},
        "termination": {"type": "delta_state", "eps": 0.1, "m_min": 1, "m_max": 8},
        "train": {"curriculum": curr} if curr else {"epochs": 1},
    })


def test_no_curriculum_section_means_cold_start():
    assert stages_from_cfg(_cfg()) == []


def test_s2_expands_to_k_stages_with_m_k_plus_1():
    st = stages_from_cfg(_cfg(max_cot_steps=3, stages=[
        {"name": "S1", "remove_cot": 0, "M": 1, "epochs": 1},
        {"expand": "S2", "epochs": 2},
        {"name": "S3", "epochs": 1},
    ]))
    assert [s.name for s in st] == ["S1", "S2.1", "S2.2", "S2.3", "S3"]
    assert [(s.remove_cot, s.M) for s in st] == [(0, 1), (1, 2), (2, 3), (3, 4), (None, None)]
    assert st[-1].is_latent_only and st[-1].max_answer_tokens == 12
    assert st[1].max_answer_tokens == 96


def test_stage_prompt_spec_keeps_everything_else():
    base = PromptSpec(max_question_tokens=640, question_truncation_side="left")
    sp = StageSpec("S2.1", 1, 2, 1, 96).prompt_spec(base)
    assert (sp.cot_keep_from, sp.max_answer_tokens, sp.max_question_tokens) == (1, 96, 640)


class _Runner:
    schedule = None
    termination = None


def test_apply_stage_fixes_m_for_training_and_eval_alike():
    """학습만 고정하고 평가를 동적으로 두면 F-031 과 같은 경로 불일치다."""
    r = _Runner(); cfg = _cfg()
    info = apply_stage(r, cfg, StageSpec("S2.2", 2, 3, 1, 96))
    assert info["M"] == 3
    assert r.schedule.sample_M() == 3
    assert type(r.termination).__name__ == "FixedMRule" and r.termination.target_m == 3


def test_apply_stage_restores_dynamic_rule_for_s3():
    r = _Runner(); cfg = _cfg()
    apply_stage(r, cfg, StageSpec("S2.1", 1, 2, 1, 96))
    info = apply_stage(r, cfg, StageSpec("S3", None, None, 1, 12))
    assert info["M"] == "dynamic"
    assert type(r.termination).__name__ == "DeltaStateRule"
    assert type(r.schedule).__name__ != "FixedSchedule"


def test_stage_m_beyond_m_max_is_rejected():
    with pytest.raises(ConfigError, match="m_max"):
        apply_stage(_Runner(), _cfg(), StageSpec("S2.9", 9, 10, 1, 96))


def test_stages_run_only_inside_phase_a():
    """Phase B 는 엔진이 얼어 있다 — 거기서 CoT 커리큘럼을 돌리면 정렬이 아니다."""
    src = (REPO_ROOT / "scripts" / "train.py").read_text(encoding="utf-8")
    assert 'stages_from_cfg(cfg) if phase.name == "A" else []' in src
