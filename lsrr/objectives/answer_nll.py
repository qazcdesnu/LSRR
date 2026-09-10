"""정답 생성 손실 L_NLL (제안서 §5).

    "h_fusion 조건 하 답 토큰들의 NLL."
"""

from __future__ import annotations

from typing import Any

import torch

from lsrr.core.errors import LSRRError
from lsrr.core.interfaces import BaseObjective
from lsrr.core.registry import OBJECTIVE_REGISTRY
from lsrr.core.types import ReasoningTrace
from lsrr.objectives.targets import loss_targets, token_nll


@OBJECTIVE_REGISTRY.register("answer_nll")
class AnswerNLL(BaseObjective):
    """최종 상태 R*에서 판독한 답의 NLL."""

    def __init__(self, **_: Any) -> None:
        super().__init__()

    def forward(
        self, trace: ReasoningTrace, batch: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        if trace.logits is None:
            raise LSRRError(
                "trace.logits가 없다. 판독 경로가 호출되지 않았거나 answer_ids가 "
                "전달되지 않았다."
            )
        loss = token_nll(trace.logits, loss_targets(batch))
        return {"loss": loss, "answer_nll": loss.detach()}


__all__ = ("AnswerNLL",)
