"""학습과 평가는 같은 것을 방출한다 (F-031).

기존 방출 테스트는 전부 `model(batch, is_eval=True)` 만 불렀다. 그래서
**학습 경로가 사고 토큰을 1개만 방출하는 것을 6개월치 실험 내내 아무도 몰랐다.**
`Trainer.forward_batch` 가 encode→refine→read 를 따로 조립하면서 궤적 수집기를
붙이지 않았고, `single` 로 학습하고 `trajectory` 로 평가하는 상태가 됐다.

손실은 정상적으로 내려갔다 — 학습 경로 안에서는 아무것도 모순되지 않기 때문이다.
신호는 **두 경로를 나란히 놓아야만** 나온다. 그래서 계약이다.

이 파일이 고정하는 것:
- 학습 경로와 평가 경로의 방출 토큰 수가 같다
- 방출 토큰 수가 실제 정제 사이클 수와 같다 (ADR-015)
- 순전파 조립 지점이 `model.forward` 하나뿐이다
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from lsrr.builder import build_slots
from lsrr.config import load_config
from lsrr.data import PromptEncoder, PromptSpec
from lsrr.data.collate import make_loader
from lsrr.model import LSRRModel
from lsrr.runtime import Trainer

pytestmark = pytest.mark.contract

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _cfg(*overrides: str):
    return load_config(
        ["exp=ablation/A_emission", "data.sizes.train=200", "device=cpu", *overrides]
    )


def _fixture(*overrides: str):
    cfg = _cfg(*overrides)
    torch.manual_seed(0)
    bundle = build_slots(cfg)
    model = LSRRModel(bundle=bundle, cfg=cfg, runner=bundle.runner)
    enc = PromptEncoder(
        bundle.encoder.tokenizer,
        PromptSpec(max_question_tokens=32, max_answer_tokens=12),
    )
    batch = next(iter(make_loader(
        bundle.data.get_split("val")[:4], enc, batch_size=4, shuffle=False
    )))
    trainer = Trainer(
        model, bundle.objective, tracker=None, cfg=cfg,
        device=torch.device("cpu"),
        params=[p for p in model.parameters() if p.requires_grad],
    )
    return model, trainer, batch


def _tokens(trace) -> int:
    """방출된 사고 토큰 수. 단일 벡터는 1개로 센다."""
    if trace.h_thought is not None:
        return int(trace.h_thought.shape[1])
    return 1


@pytest.mark.parametrize(
    "overrides,expected",
    [
        (("readout.path.emission=trajectory", "termination.type=fixed_m",
          "termination.m=4", "recurrence.schedule.type=fixed",
          "recurrence.schedule.k=4"), 4),
        (("readout.path.emission=single", "readout.fusion.anchor=ctx",
          "recurrence.schedule.type=fixed", "recurrence.schedule.k=4"), 1),
    ],
)
def test_train_and_eval_emit_the_same_number_of_tokens(overrides, expected):
    """학습이 1개를 내고 평가가 M개를 내면 두 다른 모델을 비교한 것이다 (F-031)."""
    model, trainer, batch = _fixture(*overrides)
    with torch.no_grad():
        model.train()
        train_trace = trainer.forward_batch(batch)
        model.eval()
        eval_trace = model(batch, is_eval=True)

    assert _tokens(train_trace) == expected, "학습 경로의 방출 개수가 틀렸다"
    assert _tokens(eval_trace) == expected, "평가 경로의 방출 개수가 틀렸다"


def test_emitted_tokens_match_the_refinement_cycles():
    """방출 개수 = 사이클 수. ADR-015 의 '사이클마다 토큰 하나' 그대로다."""
    model, trainer, batch = _fixture(
        "readout.path.emission=trajectory",
        "termination.type=fixed_m", "termination.m=5",
        "recurrence.schedule.type=fixed", "recurrence.schedule.k=5",
    )
    with torch.no_grad():
        model.train()
        trace = trainer.forward_batch(batch)
    assert _tokens(trace) == len(trace.per_cycle) == 5


def test_training_logits_are_not_shifted_relative_to_eval():
    """토큰 수가 같아도 정렬이 어긋나면 같은 버그의 다른 얼굴이다.

    M 을 고정하면 두 경로의 로짓 모양이 같아야 한다 — 주입 길이가 다르면
    `teacher_forced` 의 슬라이스가 달라져 모양부터 갈린다.
    """
    model, trainer, batch = _fixture(
        "readout.path.emission=trajectory",
        "termination.type=fixed_m", "termination.m=4",
        "recurrence.schedule.type=fixed", "recurrence.schedule.k=4",
    )
    with torch.no_grad():
        model.train()
        train_trace = trainer.forward_batch(batch)
        model.eval()
        eval_trace = model(batch, is_eval=True)
    assert train_trace.logits.shape == eval_trace.logits.shape
    assert train_trace.logits.shape[1] == batch["labels"].shape[1]


def test_the_trainer_does_not_assemble_its_own_forward():
    """조립 지점이 둘이면 말없이 갈라진다 — F-031 이 정확히 그랬다."""
    src = (REPO_ROOT / "lsrr" / "runtime" / "trainer.py").read_text(encoding="utf-8")
    body = src[src.index("def forward_batch"):src.index("def training_step")]
    for forbidden in ("model.refine(", "model.read(", "model.encode(", "build_memory("):
        assert forbidden not in body, (
            f"Trainer.forward_batch 가 {forbidden} 를 직접 부른다. 순전파 조립은 "
            f"model.forward 한 곳이다 (F-031)."
        )
    assert "self.model(" in body


def test_forward_is_the_only_assembly_point_in_the_package():
    """`lsrr/` 안에서 encode→refine→read 를 다시 조립하는 곳이 없어야 한다."""
    offenders = []
    for path in (REPO_ROOT / "lsrr").rglob("*.py"):
        if path.name == "model.py":
            continue
        text = path.read_text(encoding="utf-8")
        if ".refine(" in text and ".read(" in text:
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, (
        f"{offenders} 가 순전파를 따로 조립한다. 두 경로는 반드시 갈라진다 (F-031)."
    )
