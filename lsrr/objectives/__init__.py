"""학습 목표 (제안서 §5).

손실은 **값만 받는다** — 모델 모듈을 호출하지 않는다 (ADR-005).
"""

from lsrr.objectives.answer_nll import AnswerNLL
from lsrr.objectives.composite import CompositeObjective
from lsrr.objectives.targets import loss_targets, token_nll

__all__ = ("AnswerNLL", "CompositeObjective", "loss_targets", "token_nll")
