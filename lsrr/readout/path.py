"""전 사이클 공유 판독 경로 (I3, 제안서 §5).

    "풀링·융합·주입의 판독 경로는 전 사이클 공유(사이클별 헤드 금지) — 임의 시점
     종료에도 답이 판독되는 anytime 성질의 원천이다."

`R`이 최종 상태 R*이든 중간 사이클 상태 R⁽ᵐ⁾이든 **경로는 동일하다**. 이 성질을
지키려면 판독이 하나의 객체여야 하고, 모든 사이클이 그 동일 인스턴스를 호출해야
한다 (ADR-005).
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from lsrr.core.errors import AssemblyError
from lsrr.core.interfaces import BaseReadoutPath
from lsrr.core.invariants import assert_injection_space
from lsrr.core.registry import READOUT_REGISTRY
from lsrr.core.types import ContextBundle, ReadoutResult
from lsrr.readout.injection import InjectionCalibrator


@READOUT_REGISTRY.register("backbone_continuation")
class BackboneContinuationReadout(BaseReadoutPath):
    """융합 → 주입 보정 → 백본 연속 디코딩.

    학습 파라미터는 융합 헤드와 보정 게인뿐이다. 답변 생성 자체는 동결 백본이
    수행한다 (ADR-001).

    Args:
        fusion: 융합 헤드 (조립 루트가 주입).
        answer_head: `BackboneContinuation` (동결 백본 소유).
        d_in: 백본 폭. 주입 공간 검사에 쓴다 (I8).
        calibration: 주입 보정 모드 (ADR-013). 기본 `learned_rms`.
        max_new_tokens: 생성 상한.
    """

    def __init__(
        self,
        fusion: Optional[nn.Module] = None,
        answer_head: Optional[Any] = None,
        d_in: int = 768,
        calibration: str = "learned_rms",
        target_rms: Optional[float] = None,
        max_new_tokens: int = 32,
        **_: Any,
    ) -> None:
        super().__init__()
        self.fusion = fusion
        self.d_in = d_in
        self.max_new_tokens = max_new_tokens
        # 백본은 nn.Module 자식으로 등록하지 않는다 (I1)
        object.__setattr__(self, "answer_head", answer_head)

        if calibration == "none":
            self.calibrator = InjectionCalibrator(d_in=d_in, mode="none")
        elif target_rms is not None:
            self.calibrator = InjectionCalibrator(
                d_in=d_in, mode=calibration, target_rms=target_rms
            )
        elif answer_head is not None:
            self.calibrator = InjectionCalibrator.from_backbone(
                answer_head.model, mode=calibration
            )
        else:
            raise AssemblyError(
                "주입 보정 목표를 정할 수 없다. answer_head 또는 target_rms를 넘겨라 "
                "(ADR-013)."
            )

    # ------------------------------------------------------------ 융합

    def fuse(self, R: torch.Tensor, h_ctx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """R → (h_fusion, alpha). 보정까지 마친 주입 벡터를 돌려준다."""
        if self.fusion is None:
            raise AssemblyError("융합 헤드가 조립되지 않았다.")
        h_fusion, alpha = self.fusion(R, h_ctx)
        h_fusion = self.calibrator(h_fusion)
        assert_injection_space(h_fusion, self.d_in)  # I8
        return h_fusion, alpha

    # ------------------------------------------------------------ 판독

    def readout(
        self,
        R: torch.Tensor,
        h_ctx: torch.Tensor,
        context: ContextBundle,
        answer_ids: Optional[torch.Tensor] = None,
        m: Optional[int] = None,
    ) -> ReadoutResult:
        """정제 상태에서 답을 판독한다.

        `answer_ids`가 없으면 로짓을 만들지 않는다 — 융합만 필요한 호출
        (예: 진단)에서 백본을 부르지 않기 위해서다.
        """
        h_fusion, alpha = self.fuse(R, h_ctx)

        logits = None
        if answer_ids is not None:
            if self.answer_head is None:
                raise AssemblyError("연속 디코딩 헤드가 조립되지 않았다.")
            logits = self.answer_head.teacher_forced(
                h_fusion,
                context.kv_cache,
                answer_ids,
                attention_mask=context.attention_mask,
                q_len=context.q_len,
            )
        return ReadoutResult(logits=logits, h_fusion=h_fusion, alpha=alpha, m=m)

    @torch.no_grad()
    def generate(
        self,
        R: torch.Tensor,
        h_ctx: torch.Tensor,
        context: ContextBundle,
        max_new_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        """평가용 탐욕적 생성 → [B, T_gen]."""
        if self.answer_head is None:
            raise AssemblyError("연속 디코딩 헤드가 조립되지 않았다.")
        h_fusion, _ = self.fuse(R, h_ctx)
        return self.answer_head.generate(
            h_fusion,
            context.kv_cache,
            max_new_tokens=max_new_tokens or self.max_new_tokens,
            attention_mask=context.attention_mask,
            q_len=context.q_len,
        )


__all__ = ("BackboneContinuationReadout",)
