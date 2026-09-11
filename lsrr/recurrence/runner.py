"""사이클 축 제어 (제안서 §4.2 "축의 구분").

    "SSM이 스캔하는 축은 깊이(레이어) 축이고, 반복이 도는 축은 격자 밖의 정제
     사이클 m이다. L개 상태는 사이클 내에서 병렬 스캔으로 동시 갱신되며,
     사이클 간에만 순차성이 남는다."

이 파일은 **사이클 축만** 본다. 엔진 내부(`forward_step`만 호출)도, 종료 규칙
내부(`should_stop`만 호출)도 모른다.

개작: v1.0:lsrr/iteration/controller.py (3분할, ADR-005)
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch

from lsrr.core.errors import TerminationFallbackMissing
from lsrr.core.interfaces import BaseCycleRunner, CycleHook
from lsrr.core.invariants import assert_has_fallback
from lsrr.core.types import CycleDiagnostics, ReasoningTrace, TerminationSignals
from lsrr.recurrence.state import EarlyExitState
from lsrr.recurrence.tbptt import TBPTTWindow, detach_if_needed, make_window
from lsrr.termination.signals import state_delta


class CycleRunner(BaseCycleRunner):
    """정제 사이클을 돌린다.

    Args:
        engine: 레이어 축 연산자. `forward_step(R_m, R0, m)`만 호출한다.
        schedule: 학습 시 M 샘플링과 γ 가중.
        termination: 평가 시 종료 규칙. M_max 폴백을 갖는지 생성 시 검증한다 (I5).
        tbptt_k: 마지막 몇 사이클만 역전파할지.
    """

    def __init__(
        self,
        engine: Any,
        schedule: Any,
        termination: Any,
        tbptt_k: int = 4,
        **_: Any,
    ) -> None:
        self.engine = engine
        self.schedule = schedule
        self.termination = termination
        self.tbptt_k = int(tbptt_k)
        if termination is not None:
            assert_has_fallback(termination)  # I5

    # ------------------------------------------------------------ 공통

    @staticmethod
    def _dispatch(
        hooks: Sequence[CycleHook],
        m: int,
        R_m: torch.Tensor,
        R_next: torch.Tensor,
        diag: CycleDiagnostics,
    ) -> None:
        for hook in hooks:
            hook.on_cycle(m, R_m, R_next, diag)

    # ------------------------------------------------------------ 학습

    def run_train(
        self,
        R0: torch.Tensor,
        hooks: Sequence[CycleHook] = (),
        M: Optional[int] = None,
    ) -> ReasoningTrace:
        """M 샘플링 + 절단 BPTT로 학습 순전파.

        M 무작위 샘플링이 수렴 압력 (i)이다 — 임의 깊이에서 잘려도 정답을
        유지해야 하므로 정상 상태가 유리해진다 (제안서 §4.3).
        """
        M = int(M) if M is not None else self.schedule.sample_M()
        window = make_window(M, self.tbptt_k)

        trace = ReasoningTrace(R0=R0, tbptt_window=(window.start, window.M))
        R = R0
        for m in range(M):
            R = detach_if_needed(R, window, m)
            R_next = self.engine.forward_step(R, R0, m)
            per_sample = state_delta(R, R_next).detach()
            diag = CycleDiagnostics(
                m=m,
                delta_state=float(per_sample.mean()),
                # 거동 분류(게이트 ④)는 샘플별 궤적을 요구한다. 배치 평균만
                # 남기면 "문제 난이도에 따른 적응적 계산"을 사후에 볼 수 없다.
                extra={"per_sample_delta": per_sample.cpu().tolist()},
            )
            trace.per_cycle.append(diag)
            self._dispatch(hooks, m, R, R_next, diag)
            R = R_next

        trace.R_star = R
        trace.meta["M"] = M
        return trace

    # ------------------------------------------------------------ 평가

    @torch.no_grad()
    def run_eval(
        self,
        R0: torch.Tensor,
        hooks: Sequence[CycleHook] = (),
        termination: Optional[Any] = None,
        signal_fn: Optional[Any] = None,
    ) -> ReasoningTrace:
        """동적 종료로 평가 순전파.

        Args:
            signal_fn: `(R_m, R_next, m) -> TerminationSignals`. 출력 공간 규칙
                (KL·엔트로피)이 사이클별 로짓을 필요로 할 때 조립 루트가 넘긴다.
                None이면 상태 공간 Δ만 계산한다.

        m = M_max에서 반드시 정지한다 (I5).
        """
        rule = termination or self.termination
        if rule is None:
            raise TerminationFallbackMissing(
                "평가에는 종료 규칙이 필요하다. 고정 M을 원하면 "
                "termination.type=fixed_m을 쓰라."
            )

        m_max = int(rule.m_max)
        rule.reset(R0.shape[0], R0.device)
        state = EarlyExitState(R0, m_max)
        trace = ReasoningTrace(R0=R0)

        R = R0
        for m in range(m_max):
            R_next = self.engine.forward_step(R, R0, m)
            signals = (
                signal_fn(R, R_next, m)
                if signal_fn is not None
                else TerminationSignals(delta_state=state_delta(R, R_next))
            )
            stop_mask, diag = rule.should_stop(signals, m)
            diag.stopped = stop_mask.clone()

            trace.per_cycle.append(diag)
            self._dispatch(hooks, m, R, R_next, diag)

            state.update(R_next, stop_mask, m)
            R = R_next
            if state.all_done:
                break

        trace.R_star = state.R_final
        trace.stopping_cycles = state.stopping_cycles
        trace.meta["avg_cycles"] = state.avg_cycles
        trace.meta["m_max"] = m_max
        return trace


__all__ = ("CycleRunner",)
