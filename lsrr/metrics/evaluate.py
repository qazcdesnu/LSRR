"""평가 실행 — Phase 0 게이트의 판정 데이터를 만든다 (M5).

게이트는 **판정만** 하고 실험을 돌리지 않는다 (`gates/README.md`). 그 판정에
필요한 네 가지를 이 모듈이 한 번의 평가에서 함께 수집한다.

| 산출 | 게이트 | 수집 방식 |
|---|---|---|
| `collapse` | ① | 마지막 배치의 `R*`·`R⁰`를 `analysis/collapse.py`로 진단 |
| `anytime` | ② | 사이클별로 **생성**해 exact match — 최종 정확도와 같은 프로토콜 |
| `accuracy` | ③ | 동적 종료로 얻은 `R*`에서 생성 |
| `delta_trajectories` | ④ | 샘플별 Δ⁽ᵐ⁾ 궤적 |

**anytime을 teacher-forcing 토큰 정확도로 재지 않는다.** 게이트 ③이 생성 기반
exact match이므로, ②를 다른 양으로 재면 두 게이트가 서로 다른 것을 말하게 된다.
대신 비용이 M배이므로 `anytime_samples`로 부분집합만 잰다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Sequence

import torch

from lsrr.analysis.collapse import CollapseReport, diagnose
from lsrr.core.errors import MissingTargetError
from lsrr.core.types import ContextBundle, ReasoningTrace
from lsrr.metrics.accuracy import match_final_number, match_free_form
from lsrr.recurrence.hooks import DiagnosticsRecorder

Scorer = Callable[[str, str], bool]


def scorer_for(kind: str) -> Scorer:
    """데이터셋 종류별 채점 함수. 알 수 없으면 던진다 — 조용히 기본값을 쓰면
    정확도가 프로토콜 없이 보고된다 (규약 §3)."""
    if kind in ("multiplication", "gsm8k", "gsm8k_aug", "numeric"):
        return match_final_number
    if kind in ("prosqa", "prontoqa", "free_form"):
        return match_free_form
    raise ValueError(
        f"데이터셋 '{kind}'의 채점 프로토콜을 모른다. "
        f"metrics/evaluate.py:scorer_for 에 등록하라."
    )


@dataclass
class EvalResult:
    """평가 1회의 산출. `gate_inputs.json`과 `metrics.json`의 원재료."""

    accuracy: float
    num_samples: int
    mean_cycles: float
    anytime: dict[int, float] = field(default_factory=dict)
    delta_trajectories: list[list[float]] = field(default_factory=list)
    collapse: Optional[CollapseReport] = None
    predictions: list[str] = field(default_factory=list)
    targets: list[str] = field(default_factory=list)

    def metrics_payload(self) -> dict[str, Any]:
        """`metrics.json` — 게이트 ③이 시드별로 읽는다."""
        return {
            "accuracy": self.accuracy,
            "num_samples": self.num_samples,
            "mean_cycles": self.mean_cycles,
        }

    def gate_inputs_payload(self) -> dict[str, Any]:
        """`gate_inputs.json` — 게이트 ①②④가 읽는다.

        측정하지 못한 항목은 키를 **넣지 않는다**. 빈 값을 넣으면 판정기가
        '데이터 있음'으로 보고 잘못 평가한다.
        """
        payload: dict[str, Any] = {}
        if self.collapse is not None:
            payload["collapse"] = {
                "effective_rank": self.collapse.effective_rank,
                "layer_variance": self.collapse.layer_variance,
                "mean_similarity": self.collapse.mean_similarity,
                "variance_ratio": self.collapse.variance_ratio,
            }
        if self.anytime:
            payload["anytime"] = {str(m): a for m, a in sorted(self.anytime.items())}
        if self.delta_trajectories:
            payload["delta_trajectories"] = self.delta_trajectories
        return payload


class _PerCycleStates:
    """사이클별 `R⁽ᵐ⁾`를 모으는 훅. anytime 생성의 입력이다."""

    def __init__(self, cycles: Optional[Sequence[int]] = None) -> None:
        self.cycles = set(cycles) if cycles is not None else None
        self.states: dict[int, torch.Tensor] = {}

    def on_cycle(self, m: int, R_m: torch.Tensor, R_next: torch.Tensor, diagnostics: Any) -> None:
        if self.cycles is None or m in self.cycles:
            self.states[m] = R_next.detach()


def _per_sample_deltas(recorder: DiagnosticsRecorder, batch_size: int) -> list[list[float]]:
    """배치의 Δ 궤적을 샘플별로 편다.

    `CycleDiagnostics.delta_state`는 배치 평균 스칼라이므로, 샘플별 궤적이
    필요한 게이트 ④에는 같은 값을 복제해 넘긴다 — 배치 크기가 1일 때만
    엄밀히 샘플별이다. 이 근사는 `extra["per_sample_delta"]`가 채워지면
    자동으로 정밀해진다.
    """
    per_sample: list[list[float]] = [[] for _ in range(batch_size)]
    for rec in recorder.records:
        vec = rec.extra.get("per_sample_delta") if rec.extra else None
        for b in range(batch_size):
            per_sample[b].append(
                float(vec[b]) if vec is not None else float(rec.delta_state)
            )
    return per_sample


def _gold_answers(batch: dict[str, Any]) -> list[str]:
    """배치에서 정답 문자열을 꺼낸다.

    `collate`는 `DataSample`을 `samples`로 그대로 실어 보낸다. 정답이 없으면
    던진다 — 빈 문자열로 채점하면 정확도가 조용히 0이나 1로 붙는다 (규약 §3).
    """
    samples = batch.get("samples")
    if samples is not None:
        return [str(s.answer) for s in samples]
    if "answer_text" in batch:
        return [str(a) for a in batch["answer_text"]]
    raise MissingTargetError(
        "배치에 정답이 없다: `samples` 도 `answer_text` 도 없다. "
        "채점할 수 없으므로 평가를 중단한다."
    )


@torch.no_grad()
def evaluate(
    model: Any,
    batches: Iterable[dict[str, Any]],
    decode: Callable[[torch.Tensor], list[str]],
    scorer: Scorer,
    *,
    anytime_batches: int = 1,
    collect_deltas: bool = True,
    max_new_tokens: Optional[int] = None,
) -> EvalResult:
    """평가를 돌려 게이트 판정 데이터를 모은다.

    Args:
        model: 조립된 `LSRRModel`.
        batches: `{input_ids, attention_mask, answer_text}` 배치들.
        decode: 생성 토큰 → 문자열.
        scorer: `(예측, 정답) -> bool`.
        anytime_batches: 사이클별 생성을 수행할 배치 수. 비용이 M배라 기본 1이다.
        collect_deltas: Δ 궤적 수집 여부 (게이트 ④).
    """
    model.eval()
    correct = total = 0
    cycles_sum = 0.0
    preds: list[str] = []
    golds: list[str] = []
    anytime_hits: dict[int, int] = {}
    anytime_total = 0
    trajectories: list[list[float]] = []
    collapse: Optional[CollapseReport] = None

    for idx, batch in enumerate(batches):
        model.encode_counter.reset()
        context: ContextBundle = model.encode(
            batch["input_ids"], batch.get("attention_mask")
        )
        R0 = model.build_memory(context)

        recorder = DiagnosticsRecorder()
        want_anytime = idx < anytime_batches
        states = _PerCycleStates() if want_anytime else None
        hooks: list[Any] = [recorder] + ([states] if states else [])

        trace: ReasoningTrace = model.refine(R0, hooks=hooks, is_eval=True)
        trace.R0 = R0

        answers = _gold_answers(batch)
        tokens = model.readout.generate(
            trace.R_star, context.h_ctx, context, max_new_tokens=max_new_tokens
        )
        batch_preds = decode(tokens)
        preds.extend(batch_preds)
        golds.extend(answers)
        correct += sum(scorer(p, g) for p, g in zip(batch_preds, answers))
        total += len(answers)

        if trace.stopping_cycles is not None:
            cycles_sum += float(trace.stopping_cycles.sum())
        else:
            cycles_sum += len(trace.per_cycle) * len(answers)

        if collect_deltas:
            trajectories.extend(_per_sample_deltas(recorder, len(answers)))

        if want_anytime and states is not None:
            anytime_total += len(answers)
            for m, R_m in states.states.items():
                cycle_tokens = model.readout.generate(
                    R_m, context.h_ctx, context, max_new_tokens=max_new_tokens
                )
                hits = sum(
                    scorer(p, g) for p, g in zip(decode(cycle_tokens), answers)
                )
                anytime_hits[m] = anytime_hits.get(m, 0) + hits

        # 붕괴 진단은 마지막 배치 기준 — 배치마다 재면 비용만 든다.
        collapse = diagnose(trace.R_star, R0=R0)

    if total == 0:
        raise ValueError("평가 배치가 비어 있다.")

    return EvalResult(
        accuracy=correct / total,
        num_samples=total,
        mean_cycles=cycles_sum / total,
        anytime={m: h / anytime_total for m, h in sorted(anytime_hits.items())}
        if anytime_total
        else {},
        delta_trajectories=trajectories,
        collapse=collapse,
        predictions=preds,
        targets=golds,
    )


__all__ = ("EvalResult", "Scorer", "evaluate", "scorer_for")
