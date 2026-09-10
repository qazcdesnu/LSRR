"""사이클 콜백 (ADR-005).

깊은 감독은 손실 함수가 모듈 객체를 꺼내 쓰는 방식이 아니라, 훅이 **미리
판독해 값으로 넣어 주는** 방식이다. 훅은 조립 루트에서 주입된 **단일**
`ReadoutPath` 인스턴스를 쓴다 (I3).
"""

from __future__ import annotations

import random
from typing import Any, Callable, Optional, Sequence

import torch

from lsrr.core.types import ContextBundle, CycleDiagnostics, ReadoutResult
from lsrr.recurrence.tbptt import TBPTTWindow


class DiagnosticsRecorder:
    """사이클별 진단을 모은다. Δ 궤적·거동 분류의 원재료."""

    def __init__(self) -> None:
        self.records: list[CycleDiagnostics] = []

    def on_cycle(
        self,
        m: int,
        R_m: torch.Tensor,
        R_next: torch.Tensor,
        diagnostics: CycleDiagnostics,
    ) -> None:
        self.records.append(diagnostics)

    def deltas(self) -> list[float]:
        return [d.delta_state for d in self.records]


class ReadoutHook:
    """지정된 사이클에서 판독 경로를 호출해 결과를 값으로 축적한다.

    깊은 감독(학습)과 anytime 곡선(평가)이 같은 훅을 쓴다 — 판독 경로가 전
    사이클 공유이므로(I3) 둘을 나눌 이유가 없다.

    Args:
        readout: **단일** `ReadoutPath` 인스턴스. 매 사이클 같은 객체를 쓴다.
        context: 인코딩 산출물 (h_ctx·KV 재사용).
        answer_ids: teacher-forcing 입력. None이면 로짓을 만들지 않는다.
        cycles: 판독할 사이클 인덱스 집합. None이면 전 사이클.
    """

    def __init__(
        self,
        readout: Any,
        context: ContextBundle,
        answer_ids: Optional[torch.Tensor] = None,
        cycles: Optional[Sequence[int]] = None,
    ) -> None:
        self.readout = readout
        self.context = context
        self.answer_ids = answer_ids
        self.cycles = set(cycles) if cycles is not None else None
        self.results: list[ReadoutResult] = []
        self.instances: list[Any] = []  # I3 검증용

    def on_cycle(
        self,
        m: int,
        R_m: torch.Tensor,
        R_next: torch.Tensor,
        diagnostics: CycleDiagnostics,
    ) -> None:
        if self.cycles is not None and m not in self.cycles:
            return
        self.instances.append(self.readout)
        self.results.append(
            self.readout.readout(
                R_next,
                self.context.h_ctx,
                self.context,
                answer_ids=self.answer_ids,
                m=m,
            )
        )

    def supervised_cycles(self) -> list[int]:
        return [r.m for r in self.results if r.m is not None]


def sample_supervision_cycles(
    window: TBPTTWindow,
    num_cycles: int = 2,
    rng: Optional[random.Random] = None,
) -> list[int]:
    """깊은 감독 집합 S를 고른다 (제안서 §5, ADR-006).

        "감독 집합 S는 truncated BPTT 윈도 내에서 1~2개 사이클을 균등 샘플링
         (윈도 밖은 detach로 감독 효과 없음)"

    마지막 사이클은 L_NLL이 이미 감독하므로 후보에서 제외한다 — 포함하면 같은
    항을 두 번 세게 된다.
    """
    candidates = [m for m in window.cycles() if m < window.M - 1]
    if not candidates:
        return []
    r = rng or random
    return sorted(r.sample(candidates, min(num_cycles, len(candidates))))


__all__ = ("DiagnosticsRecorder", "ReadoutHook", "sample_supervision_cycles")
