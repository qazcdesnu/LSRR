"""조립된 LSRR 모델의 순전파 계약.

이 파일은 어떤 연산도 직접 구현하지 않는다 — 슬롯 인터페이스만 호출한다.
연산이 여기 들어오려 하면 그것은 어느 하위 패키지의 역할인지 다시 물을 신호다.

파이프라인 (ARCHITECTURE.md §2):
    질문 → [백본 1회] → memory → recurrence×engine → readout → 백본 연속 디코딩
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch
import torch.nn as nn
from omegaconf import DictConfig

from lsrr.builder import SlotBundle
from lsrr.config.schema import get_path
from lsrr.core.errors import AssemblyError
from lsrr.core.interfaces import BaseCycleRunner, CycleHook
from lsrr.core.invariants import EncodeCounter, assert_injection_space
from lsrr.core.types import ContextBundle, ReasoningTrace


class _TrajectoryCollector:
    """사이클별 정제 상태를 모은다 — 궤적 방출의 입력 (ADR-015).

    훅으로 받는 이유는 `recurrence` 가 방출 구조를 몰라야 하기 때문이다.
    사이클 축과 판독 축의 분리를 유지한다 (ADR-005).
    """

    def __init__(self) -> None:
        self.states: list[torch.Tensor] = []

    def on_cycle(
        self, m: int, R_m: torch.Tensor, R_next: torch.Tensor, diagnostics: Any
    ) -> None:
        self.states.append(R_next)


class LSRRModel(nn.Module):
    """Layer-State Recurrent Reasoner.

    학습 대상은 어댑터 + 엔진 + 융합/판독 헤드뿐이다 (제안서 §5).
    인코더(백본)는 동결이며 `nn.Module` 자식으로 등록하지 않는다 — 체크포인트에
    백본 가중치가 섞이는 것을 구조적으로 막기 위해서다.
    """

    def __init__(
        self,
        bundle: SlotBundle,
        cfg: Optional[DictConfig] = None,
        runner: Optional[BaseCycleRunner] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.widths = dict(bundle.widths)
        self.warnings = list(bundle.warnings)
        self.slot_device = bundle.device

        # 동결 백본: nn.Module 자식이 아니다 (체크포인트 오염 방지, I1)
        object.__setattr__(self, "_encoder", bundle.encoder)
        object.__setattr__(self, "_data", bundle.data)

        # 학습 대상 슬롯
        self.pooler = bundle.pooler
        self.pipeline = bundle.pipeline
        self.composer = bundle.composer
        self.scope = bundle.scope
        self.adapter = bundle.adapter
        self.engine = bundle.engine
        self.fusion = bundle.fusion
        self.readout = bundle.readout
        self.objective = bundle.objective

        # 비학습 정책 객체
        object.__setattr__(self, "schedule", bundle.schedule)
        object.__setattr__(self, "termination", bundle.termination)
        object.__setattr__(self, "stability", bundle.stability)
        object.__setattr__(self, "runner", runner)

        allow_reencode = bool(
            get_path(cfg, "experimental.reencoding_loop", False) if cfg else False
        )
        self.encode_counter = EncodeCounter(allow_reencoding=allow_reencode)

    # ------------------------------------------------------------ 접근자

    @property
    def encoder(self) -> Any:
        return self._encoder

    @property
    def d_in(self) -> int:
        return int(self.widths["d_in"])

    @property
    def d_model(self) -> int:
        return int(self.widths["d_model"])

    def trainable_parameters(self) -> list[nn.Parameter]:
        """옵티마이저에 넘길 파라미터. 백본은 여기 없다 (§5)."""
        return [p for p in self.parameters() if p.requires_grad]

    def parameter_report(self) -> dict[str, Any]:
        """학습 파라미터 수와 백본 대비 비율.

        제안서 §5의 "백본 대비 약 3% 이내" 주장이 매 런에서 확인되어야 한다.
        """
        trainable = sum(p.numel() for p in self.trainable_parameters())
        backbone = 0
        enc = self._encoder
        if enc is not None and hasattr(enc, "num_parameters"):
            backbone = int(enc.num_parameters())
        return {
            "trainable": trainable,
            "backbone": backbone,
            "ratio": (trainable / backbone) if backbone else None,
        }

    # ------------------------------------------------------------ 단계

    def encode(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> ContextBundle:
        """백본 1회 인코딩 (I2). 사이클 루프 안에서 호출하지 않는다."""
        if self._encoder is None:
            raise AssemblyError("인코더 없이 encode()를 호출했다.")
        self.encode_counter.record()
        return self.pool_context(self._encoder.encode(input_ids, attention_mask))

    def pool_context(self, context: ContextBundle) -> ContextBundle:
        """질문 전체 풀링으로 글로벌 문맥 보완항을 채운다 (제안서 §4.1, ADR-004).

        풀러는 학습 대상이므로 동결 세션 안이 아니라 모델의 자식으로 등록되어
        있고, 따라서 인코딩 직후 여기서 호출한다 (ADR-012).
        """
        if self.pooler is None or context.hidden_stack is None:
            return context
        context.H_pool = self.pooler(context.hidden_stack, context.attention_mask)
        return context

    def build_memory(self, context: ContextBundle) -> torch.Tensor:
        """ContextBundle → R⁰. 순서 고정: compose → scope → adapt (memory/README)."""
        if self.pipeline is not None:
            return self.pipeline(context)
        H = context.H_last
        if self.composer is not None:
            H = self.composer(H, context.H_pool)
        if self.scope is not None:
            H = self.scope(H)
        if self.adapter is not None:
            H = self.adapter(H)
        return H

    def refine(
        self,
        R0: torch.Tensor,
        hooks: Sequence[CycleHook] = (),
        is_eval: bool = False,
        M: Optional[int] = None,
    ) -> ReasoningTrace:
        """사이클 축 반복. 축 제어는 전부 runner의 일이다.

        `M`을 넘기면 스케줄 샘플링을 건너뛴다. 깊은 감독이 판독 훅을 만들기 전에
        TBPTT 윈도를 알아야 하므로, 호출자가 M을 먼저 뽑아 쓰는 경로다 (ADR-006).
        평가 경로에서는 무시된다 — 그쪽은 종료 규칙이 M을 정한다.
        """
        if self.runner is None:
            raise AssemblyError(
                "CycleRunner가 조립되지 않았다. `lsrr.recurrence`는 M4에서 구현된다 "
                "— 그때까지는 runner를 직접 주입하라 (ROADMAP.md)."
            )
        return (
            self.runner.run_eval(R0, hooks=hooks)
            if is_eval
            else self.runner.run_train(R0, hooks=hooks, M=M)
        )

    @property
    def _emits_trajectory(self) -> bool:
        """판독 경로가 궤적을 방출하는가 (Ablation A 의 축)."""
        return getattr(self.readout, "emission", "single") == "trajectory"

    def read(
        self,
        R: torch.Tensor,
        context: ContextBundle,
        answer_ids: Optional[torch.Tensor] = None,
        m: Optional[int] = None,
        prefix: Optional[Sequence[torch.Tensor]] = None,
    ) -> Any:
        """전 사이클 공유 판독 경로 호출 (I3).

        `prefix` 를 주면 `[*prefix, R]` 를 궤적으로 방출한다 (ADR-015).
        """
        if self.readout is None:
            raise AssemblyError("판독 경로가 조립되지 않았다.")
        result = self.readout.readout(
            R, context.h_ctx, context, answer_ids=answer_ids, m=m, prefix=prefix
        )
        if result.h_thought is not None:
            assert_injection_space(result.h_thought, self.d_in)  # I8 (시퀀스)
        elif result.h_fusion is not None:
            assert_injection_space(result.h_fusion, self.d_in)  # I8
        return result

    # ------------------------------------------------------------ 순전파

    @property
    def _needs_per_cycle_readout(self) -> bool:
        """사이클별 판독이 필요한가 — anytime 곡선(②) 또는 깊은 감독(ADR-006)."""
        return bool(
            get_path(self.cfg, "recurrence.hooks.readout_per_cycle", False)
            or get_path(self.cfg, "objective.deep_supervision.enabled", False)
        )

    def rollout(
        self,
        batch: dict[str, Any],
        is_eval: bool = False,
        hooks: Sequence[CycleHook] = (),
        M: Optional[int] = None,
        readout_cycles: Optional[Sequence[int]] = None,
    ) -> tuple[ContextBundle, ReasoningTrace, list[torch.Tensor]]:
        """순전파의 **판독 앞 절반** — encode → build_memory → refine → 상태 수집.

        판독은 용도마다 다르다: 학습·NLL 은 teacher forcing(`forward`), 평가는
        탐욕적 생성(`metrics/evaluate.py`). 하지만 **그 앞 절반과 방출 규칙은
        같아야 한다.** 각자 조립하면 갈라진다 — 실제로 두 번 갈라졌다 (F-031):
        학습이 궤적 수집기를 빠뜨려 토큰 1개만 방출했고, 평가는 `prefix` 를
        넘기지 않아 정확도를 단일 토큰 생성으로 쟀다.

        Returns:
            `(context, trace, states)`. `states` 는 사이클 순서의 `R⁽¹⁾…R⁽ᴹ⁾` 이며
            궤적 방출이 아니면 빈 목록이다. 접두는 `emission_prefix` 로 만든다.
        """
        self.encode_counter.reset()

        context = self.encode(batch["input_ids"], batch.get("attention_mask"))
        R0 = self.build_memory(context)

        # 궤적 방출이면 사이클별 상태를 모아야 한다 (ADR-015). 훅으로 모으므로
        # runner 는 방출 구조를 모른다 — 축의 분리를 유지한다.
        collector = _TrajectoryCollector() if self._emits_trajectory else None
        readout_hook = None
        if self._needs_per_cycle_readout:
            from lsrr.recurrence.hooks import ReadoutHook

            readout_hook = ReadoutHook(
                readout=self.readout,
                context=context,
                answer_ids=batch.get("target_ids"),
                cycles=readout_cycles,
            )

        all_hooks: list[Any] = list(hooks)
        if readout_hook is not None:
            all_hooks.append(readout_hook)
        if collector is not None:
            all_hooks.append(collector)

        trace = self.refine(R0, hooks=all_hooks, is_eval=is_eval, M=M)
        trace.R0 = R0
        trace.h_ctx = context.h_ctx
        if readout_hook is not None:
            trace.per_cycle_readout = readout_hook.results
            trace.supervised_cycles = readout_hook.supervised_cycles()
        trace.meta.setdefault("encode_count", self.encode_counter.count)

        states = list(collector.states) if collector is not None else []
        return context, trace, states

    def emission_prefix(
        self, states: Sequence[torch.Tensor], upto: Optional[int] = None
    ) -> Optional[list[torch.Tensor]]:
        """방출 접두 — **방출 구조를 해석하는 유일한 곳이다.**

        Args:
            states: `rollout` 이 돌려준 사이클 상태.
            upto: 여기까지의 상태만 접두로 쓴다. anytime 곡선이 "사이클 m 에서
                멈췄다면" 을 물을 때 쓴다. None 이면 마지막 직전까지 — 마지막
                상태는 호출부가 `R` 로 넘기기 때문이다.

        `single` 방출이면 항상 None 이다. 접두가 없으면 토큰 하나만 주입된다.
        """
        if not self._emits_trajectory or not states:
            return None
        cut = len(states) - 1 if upto is None else int(upto)
        head = list(states[:cut])
        return head or None

    def forward(
        self,
        batch: dict[str, Any],
        is_eval: bool = False,
        hooks: Sequence[CycleHook] = (),
        M: Optional[int] = None,
        readout_cycles: Optional[Sequence[int]] = None,
    ) -> ReasoningTrace:
        """전체 순전파 — **학습과 평가가 공유하는 유일한 경로다.**

        호출부가 순전파를 따로 조립하면 두 경로가 말없이 갈라진다. 실제로
        그랬다 (F-031): 학습이 궤적 수집기를 붙이지 않아 사고 토큰을 1개만
        방출하고, 평가는 M개를 방출했다 — `single` 로 학습하고 `trajectory` 로
        평가한 셈이다. 그러므로 **여기 말고 다른 곳에서 encode→refine→read 를
        조립하지 않는다.**

        Args:
            batch: 최소 `input_ids`. 학습 시 `labels`/`target_ids`.
            is_eval: 동적 종료를 쓰는 평가 모드.
            hooks: 추가 사이클 콜백 (진단 기록 등). 궤적 수집기와 판독 훅은
                이 메서드가 직접 붙인다 — 붙이는 것을 잊는 것이 곧 F-031 이다.
            M: 사이클 수를 고정한다. 깊은 감독이 TBPTT 윈도를 먼저 알아야 할 때
                호출자가 뽑아 넘긴다 (ADR-006). 평가 경로에서는 무시된다.
            readout_cycles: 판독할 사이클 인덱스. None 이면 전 사이클.

        Returns:
            ReasoningTrace. 모듈 객체는 담지 않는다 (ADR-005).
        """
        context, trace, states = self.rollout(
            batch, is_eval=is_eval, hooks=hooks, M=M, readout_cycles=readout_cycles
        )
        R_final = trace.R_star if trace.R_star is not None else trace.R0

        readout = self.read(
            R_final,
            context,
            answer_ids=batch.get("target_ids"),
            prefix=self.emission_prefix(states),
        )
        trace.logits = readout.logits
        trace.h_fusion = readout.h_fusion
        trace.h_thought = readout.h_thought
        trace.alpha = readout.alpha
        return trace

    def __repr__(self) -> str:
        parts = [
            f"{k}={type(v).__name__}"
            for k, v in (
                ("encoder", self._encoder),
                ("adapter", self.adapter),
                ("engine", self.engine),
                ("termination", self.termination),
                ("readout", self.readout),
            )
            if v is not None
        ]
        return f"LSRRModel(d_in={self.widths.get('d_in')}, d_model={self.widths.get('d_model')}, {', '.join(parts)})"


__all__ = ("LSRRModel",)
