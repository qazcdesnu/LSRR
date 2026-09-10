"""수렴 기반 자율 종료 (제안서 §4.3)."""

from lsrr.termination.base import TerminationRuleBase
from lsrr.termination.rules import (
    DeltaStateRule,
    EntropyOutputRule,
    FixedMRule,
    KLOutputRule,
)
from lsrr.termination.signals import (
    output_entropy,
    output_kl,
    relative_delta,
    state_delta,
)

__all__ = (
    "TerminationRuleBase",
    "FixedMRule",
    "DeltaStateRule",
    "KLOutputRule",
    "EntropyOutputRule",
    "state_delta",
    "output_kl",
    "output_entropy",
    "relative_delta",
)
