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
    ) -> ReasoningTrace:
        """사이클 축 반복. 축 제어는 전부 runner의 일이다."""
        if self.runner is None:
            raise AssemblyError(
                "CycleRunner가 조립되지 않았다. `lsrr.recurrence`는 M4에서 구현된다 "
                "— 그때까지는 runner를 직접 주입하라 (ROADMAP.md)."
            )
        return (
            self.runner.run_eval(R0, hooks=hooks)
            if is_eval
            else self.runner.run_train(R0, hooks=hooks)
        )

    def read(
        self,
        R: torch.Tensor,
        context: ContextBundle,
        answer_ids: Optional[torch.Tensor] = None,
        m: Optional[int] = None,
    ) -> Any:
        """전 사이클 공유 판독 경로 호출 (I3)."""
        if self.readout is None:
            raise AssemblyError("판독 경로가 조립되지 않았다.")
        result = self.readout.readout(
            R, context.h_ctx, context, answer_ids=answer_ids, m=m
        )
        if result.h_fusion is not None:
            assert_injection_space(result.h_fusion, self.d_in)  # I8
        return result

    # ------------------------------------------------------------ 순전파

    def forward(
        self,
        batch: dict[str, Any],
        is_eval: bool = False,
        hooks: Sequence[CycleHook] = (),
    ) -> ReasoningTrace:
        """전체 순전파.

        Args:
            batch: 최소 `input_ids`. 학습 시 `labels`/`target_ids`.
            is_eval: 동적 종료를 쓰는 평가 모드.
            hooks: 사이클 콜백 (진단 기록·깊은 감독·anytime 곡선).

        Returns:
            ReasoningTrace. 모듈 객체는 담지 않는다 (ADR-005).
        """
        self.encode_counter.reset()

        context = self.encode(batch["input_ids"], batch.get("attention_mask"))
        R0 = self.build_memory(context)

        trace = self.refine(R0, hooks=hooks, is_eval=is_eval)
        trace.R0 = R0
        trace.h_ctx = context.h_ctx

        readout = self.read(
            trace.R_star if trace.R_star is not None else R0,
            context,
            answer_ids=batch.get("target_ids"),
        )
        trace.logits = readout.logits
        trace.h_fusion = readout.h_fusion
        trace.alpha = readout.alpha
        trace.meta.setdefault("encode_count", self.encode_counter.count)
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
