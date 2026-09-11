"""평가 경로 — 게이트 판정 데이터 산출 (M5).

게이트는 판정만 하고 실험을 돌리지 않으므로, **판정 데이터를 만드는 쪽**이
정확해야 한다. 이 파일이 고정하는 것:

- 측정하지 못한 항목은 `gate_inputs.json` 에 키를 넣지 않는다 (빈 값 = 데이터 있음 으로 오독)
- anytime 은 최종 정확도와 **같은 프로토콜**(생성 기반 exact match)로 잰다
- 정답이 없는 배치는 조용히 넘어가지 않는다
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from lsrr.core.errors import MissingTargetError
from lsrr.core.types import CycleDiagnostics, ReasoningTrace
from lsrr.metrics.evaluate import EvalResult, evaluate, scorer_for
from lsrr.recurrence.hooks import DiagnosticsRecorder


class _Sample:
    def __init__(self, answer: str) -> None:
        self.answer = answer


class _Ctx:
    h_ctx = None
    kv_cache = None
    attention_mask = None
    q_len = 0


class _Counter:
    def reset(self) -> None:
        self.count = 0


class _Readout:
    """사이클별로 다른 답을 내는 스텁 — anytime 곡선이 실제로 갈리는지 본다."""

    def __init__(self, per_cycle: dict[int, list[str]], final: list[str]) -> None:
        self.per_cycle = per_cycle
        self.final = final
        self.calls: list[Any] = []
        self.injected: list[int] = []

    def generate(self, R, h_ctx, context, max_new_tokens=None, prefix=None):
        tag = int(R[0, 0, 0].item())
        # 주입되는 사고 토큰 수 = 접두 + 마지막 상태. 궤적 생성인지 단일 생성인지
        # 여기서 기록해야 "정확도를 단일 토큰으로 쟀다" 를 테스트가 잡는다 (F-031).
        self.calls.append(tag)
        self.injected.append(len(prefix or []) + 1)
        texts = self.per_cycle.get(tag, self.final)
        return torch.tensor([[hash(t) % 100] for t in texts])


class _Model:
    """조립 없이 평가 흐름만 태우는 최소 스텁."""

    def __init__(self, readout: _Readout, M: int = 3, batch_shapes=(2,),
                 emits_trajectory: bool = True) -> None:
        self.readout = readout
        self.encode_counter = _Counter()
        self.M = M
        self.emits_trajectory = emits_trajectory

    def eval(self):
        return self

    # `LSRRModel` 과 같은 계약을 스텁도 지켜야 한다 — 평가는 앞 절반을 모델에
    # 맡기고 방출 해석도 모델에 묻는다 (F-031).
    def rollout(self, batch, is_eval=False, hooks=(), M=None, readout_cycles=None):
        self.encode_counter.reset()
        context = self.encode(batch["input_ids"], batch.get("attention_mask"))
        R0 = self.build_memory(context)
        states: list[torch.Tensor] = []

        class _Collect:
            def on_cycle(self, m, R_m, R_next, diagnostics):
                states.append(R_next)

        trace = self.refine(R0, hooks=[*hooks, _Collect()], is_eval=is_eval)
        trace.R0 = R0
        return context, trace, (states if self.emits_trajectory else [])

    def emission_prefix(self, states, upto=None):
        if not self.emits_trajectory or not states:
            return None
        cut = len(states) - 1 if upto is None else int(upto)
        return list(states[:cut]) or None

    def encode(self, input_ids, attention_mask=None):
        return _Ctx()

    def build_memory(self, context):
        return torch.full((2, 4, 8), -1.0)

    def refine(self, R0, hooks=(), is_eval=False):
        trace = ReasoningTrace(R0=R0)
        R = R0
        for m in range(self.M):
            R_next = torch.full_like(R0, float(m))
            diag = CycleDiagnostics(
                m=m,
                delta_state=1.0 / (m + 1),
                extra={"per_sample_delta": [1.0 / (m + 1), 2.0 / (m + 1)]},
            )
            trace.per_cycle.append(diag)
            for h in hooks:
                h.on_cycle(m, R, R_next, diag)
            R = R_next
        trace.R_star = R
        trace.stopping_cycles = torch.tensor([self.M, self.M])
        return trace


def _decode(tokens: torch.Tensor) -> list[str]:
    return [str(int(t[0])) for t in tokens]


def _batches(n: int = 1):
    for _ in range(n):
        yield {
            "input_ids": torch.zeros(2, 4, dtype=torch.long),
            "attention_mask": torch.ones(2, 4, dtype=torch.long),
            "samples": [_Sample("42"), _Sample("7")],
        }


# ---------------------------------------------------------------- 채점 프로토콜

def test_scorer_is_chosen_by_dataset():
    assert scorer_for("multiplication")("the answer is 42", "42")
    assert scorer_for("prosqa")("sterpus", "Sally is a sterpus.")


def test_unknown_dataset_is_fatal():
    """조용히 기본값을 쓰면 정확도가 프로토콜 없이 보고된다."""
    with pytest.raises(ValueError, match="채점 프로토콜을 모른다"):
        scorer_for("nonexistent_dataset")


# ---------------------------------------------------------------- 정답 부재

def test_batch_without_answers_is_fatal():
    model = _Model(_Readout({}, ["42", "7"]))
    bad = [{"input_ids": torch.zeros(2, 4, dtype=torch.long), "attention_mask": None}]
    with pytest.raises(MissingTargetError, match="정답이 없다"):
        evaluate(model, bad, decode=_decode, scorer=lambda p, g: True)


def test_empty_evaluation_is_fatal():
    model = _Model(_Readout({}, []))
    with pytest.raises(ValueError, match="비어 있다"):
        evaluate(model, [], decode=_decode, scorer=lambda p, g: True)


# ---------------------------------------------------------------- 산출 스키마

def test_gate_inputs_omit_unmeasured_keys():
    """빈 값을 넣으면 판정기가 '데이터 있음'으로 보고 잘못 평가한다."""
    empty = EvalResult(accuracy=0.5, num_samples=2, mean_cycles=3.0)
    payload = empty.gate_inputs_payload()
    assert payload == {}
    for key in ("collapse", "anytime", "delta_trajectories"):
        assert key not in payload


def test_gate_inputs_include_what_was_measured():
    model = _Model(_Readout({}, ["42", "7"]))
    result = evaluate(
        model, _batches(1), decode=_decode,
        scorer=lambda p, g: True, anytime_batches=1,
    )
    payload = result.gate_inputs_payload()
    assert set(payload) == {"collapse", "anytime", "delta_trajectories"}
    assert payload["collapse"]["effective_rank"] > 0


def test_metrics_payload_carries_gate3_input():
    model = _Model(_Readout({}, ["42", "99"]))
    result = evaluate(
        model, _batches(1), decode=_decode, scorer=lambda p, g: p == g
    )
    payload = result.metrics_payload()
    assert set(payload) == {"accuracy", "num_samples", "mean_cycles"}
    assert payload["num_samples"] == 2


# ---------------------------------------------------------------- Δ 궤적 (④)

def test_delta_trajectories_are_per_sample():
    """배치 평균만 남기면 거동 분류가 샘플별 적응을 볼 수 없다."""
    model = _Model(_Readout({}, ["42", "7"]))
    result = evaluate(model, _batches(1), decode=_decode, scorer=lambda p, g: True)
    assert len(result.delta_trajectories) == 2
    assert result.delta_trajectories[0] != result.delta_trajectories[1]


def test_delta_trajectories_fall_back_to_batch_mean():
    """샘플별 Δ 가 없는 진단이라도 궤적 자체는 나와야 한다."""
    recorder = DiagnosticsRecorder()
    for m in range(3):
        recorder.records.append(CycleDiagnostics(m=m, delta_state=1.0 / (m + 1)))
    from lsrr.metrics.evaluate import _per_sample_deltas

    traj = _per_sample_deltas(recorder, batch_size=2)
    assert traj[0] == traj[1] == [1.0, 0.5, pytest.approx(1 / 3)]


# ---------------------------------------------------------------- anytime (②)

def test_anytime_uses_the_same_protocol_as_final_accuracy():
    """사이클별로 생성해 exact match 한다 — teacher-forcing 토큰 정확도가 아니다."""
    readout = _Readout(per_cycle={0: ["1", "1"], 1: ["42", "1"], 2: ["42", "7"]},
                       final=["42", "7"])
    model = _Model(readout, M=3)
    result = evaluate(
        model, _batches(1), decode=_decode,
        scorer=lambda p, g: p == str(hash(g) % 100), anytime_batches=1,
    )
    assert set(result.anytime) == {0, 1, 2}
    # 사이클이 갈수록 좋아지는 스텁이므로 곡선이 올라야 한다.
    assert result.anytime[0] <= result.anytime[2]
    # "사이클 m 에서 멈췄다면" = 궤적의 앞 m+1개 토큰. 전부 1이면 방출 구조가
    # 아니라 마지막 상태의 품질을 잰 것이다 (F-031).
    assert readout.injected == [3, 1, 2, 3]  # 최종(M=3) + 사이클 0·1·2


def test_anytime_is_skipped_when_budget_is_zero():
    """비용이 M배라 끄고 돌릴 수 있어야 한다."""
    model = _Model(_Readout({}, ["42", "7"]))
    result = evaluate(
        model, _batches(1), decode=_decode,
        scorer=lambda p, g: True, anytime_batches=0,
    )
    assert result.anytime == {}
    assert "anytime" not in result.gate_inputs_payload()


def test_anytime_costs_m_generations_per_batch():
    readout = _Readout({}, ["42", "7"])
    model = _Model(readout, M=3)
    evaluate(model, _batches(1), decode=_decode, scorer=lambda p, g: True,
             anytime_batches=1)
    # 최종 1회 + 사이클 3회
    assert len(readout.calls) == 4


def test_final_accuracy_is_generated_from_the_whole_trajectory():
    """정확도를 단일 토큰 생성으로 재면 Ablation A 전체가 무의미해진다 (F-031)."""
    readout = _Readout({}, ["42", "7"])
    model = _Model(readout, M=4)
    evaluate(model, _batches(1), decode=_decode, scorer=lambda p, g: True,
             anytime_batches=0)
    assert readout.injected == [4], "최종 생성이 궤적 전체를 주입하지 않았다"


def test_single_emission_injects_one_token():
    """`single` 조건은 대조군이다 — 여기서 궤적이 새면 절제가 무의미해진다."""
    readout = _Readout({}, ["42", "7"])
    model = _Model(readout, M=4, emits_trajectory=False)
    evaluate(model, _batches(1), decode=_decode, scorer=lambda p, g: True,
             anytime_batches=0)
    assert readout.injected == [1]
