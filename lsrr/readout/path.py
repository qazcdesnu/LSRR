"""전 사이클 공유 판독 경로 (I3, 제안서 §5).

    "풀링·융합·주입의 판독 경로는 전 사이클 공유(사이클별 헤드 금지) — 임의 시점
     종료에도 답이 판독되는 anytime 성질의 원천이다."

`R`이 최종 상태 R*이든 중간 사이클 상태 R⁽ᵐ⁾이든 **경로는 동일하다**. 이 성질을
지키려면 판독이 하나의 객체여야 하고, 모든 사이클이 그 동일 인스턴스를 호출해야
한다 (ADR-005).
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

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

    def fuse(
        self,
        R: torch.Tensor,
        h_ctx: torch.Tensor,
        use_anchor: Optional[bool] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """R → (h_fusion, alpha). 보정까지 마친 주입 벡터를 돌려준다.

        `use_anchor` 는 `anchor="first"` 에서 사이클별로 갈린다 (ADR-015).
        """
        if self.fusion is None:
            raise AssemblyError("융합 헤드가 조립되지 않았다.")
        h_fusion, alpha = self.fusion(R, h_ctx, use_anchor=use_anchor)
        h_fusion = self.calibrator(h_fusion)
        # 안정화 램프는 보정 뒤에 — 앞에 걸면 RMS 정규화가 소거한다.
        post = getattr(self.fusion, "post_gate", None)
        if post is not None:
            h_fusion = post(h_fusion)
        assert_injection_space(h_fusion, self.d_in)  # I8
        return h_fusion, alpha

    def emit_trajectory(
        self, states: Sequence[torch.Tensor], h_ctx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """사이클별 상태들 → 잠재 사고 토큰 시퀀스 (v2.1 §4.4, ADR-015).

            h⁽ᵐ⁾ = calib(W_r · pool(R⁽ᵐ⁾) + h_ctx),   m = 1..M

        Args:
            states: `M`개의 `[B, L, d_model]`. 사이클 순서여야 한다.
            h_ctx: `[B, d_in]` 어댑터 통과 전 백본 원본 h⁽ᴸ⁾ (ADR-003).

        Returns:
            `(H_thought [B, M, d_in], alpha [B, M, L])`

        **같은 `fuse()` 를 M 번 부른다.** 사이클별 헤드를 두면 I3 이 깨지고
        anytime 성질과 깊은 감독이 동시에 무너진다. v1 단일 벡터는 `M=1` 인
        특수 경우이므로 별도 경로를 두지 않는다 (ADR-009).
        """
        if not states:
            raise AssemblyError(
                "방출할 사이클 상태가 없다. M_min ≥ 1 이어야 한다 (v2.1 §4.3)."
            )
        fused, alphas = [], []
        for i, R in enumerate(states):
            # anchor="first" 는 첫 토큰만 앵커를 섞는다. 나머지 모드는 이 인자를
            # 무시하므로 분기를 여기 한 곳에만 둔다.
            h, a = self.fuse(R, h_ctx, use_anchor=(i == 0))
            fused.append(h)
            alphas.append(a)
        trajectory = torch.stack(fused, dim=1)  # [B, M, d_in]
        assert_injection_space(trajectory, self.d_in)  # I8 (시퀀스)
        return trajectory, torch.stack(alphas, dim=1)

    # ------------------------------------------------------------ 판독

    def readout(
        self,
        R: torch.Tensor,
        h_ctx: torch.Tensor,
        context: ContextBundle,
        answer_ids: Optional[torch.Tensor] = None,
        m: Optional[int] = None,
        prefix: Optional[Sequence[torch.Tensor]] = None,
    ) -> ReadoutResult:
        """정제 상태에서 답을 판독한다.

        `answer_ids`가 없으면 로짓을 만들지 않는다 — 융합만 필요한 호출
        (예: 진단)에서 백본을 부르지 않기 위해서다.

        Args:
            prefix: 주면 `[*prefix, R]` 을 궤적으로 방출해 주입한다. v2.1 §5.2 의
                `L_DeepSup` 이 요구하는 **궤적 접두 조건부 판독**
                (`P(y | X, h⁽¹⁾..h⁽ᵐ⁾)`) 이 이 인자로 표현된다.
                None 이면 `R` 하나만 주입한다 (v1 거동).
        """
        states = [*prefix, R] if prefix else [R]
        trajectory, alphas = self.emit_trajectory(states, h_ctx)
        h_fusion = trajectory[:, -1]
        alpha = alphas[:, -1]

        logits = None
        if answer_ids is not None:
            if self.answer_head is None:
                raise AssemblyError("연속 디코딩 헤드가 조립되지 않았다.")
            logits = self.answer_head.teacher_forced(
                trajectory,
                context.kv_cache,
                answer_ids,
                attention_mask=context.attention_mask,
                q_len=context.q_len,
            )
        return ReadoutResult(
            logits=logits,
            h_fusion=h_fusion,
            alpha=alpha,
            m=m,
            h_thought=trajectory,
        )

    @torch.no_grad()
    def generate(
        self,
        R: torch.Tensor,
        h_ctx: torch.Tensor,
        context: ContextBundle,
        max_new_tokens: Optional[int] = None,
        prefix: Optional[Sequence[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """평가용 탐욕적 생성 → [B, T_gen].

        `prefix` 를 주면 궤적 `[*prefix, R]` 전체를 주입한다 (v2.1 §4.4).
        """
        if self.answer_head is None:
            raise AssemblyError("연속 디코딩 헤드가 조립되지 않았다.")
        states = [*prefix, R] if prefix else [R]
        trajectory, _ = self.emit_trajectory(states, h_ctx)
        return self.answer_head.generate(
            trajectory,
            context.kv_cache,
            max_new_tokens=max_new_tokens or self.max_new_tokens,
            attention_mask=context.attention_mask,
            q_len=context.q_len,
        )


__all__ = ("BackboneContinuationReadout",)
