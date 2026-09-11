"""단계별 게이트 판정 (제안서 §6.3, ADR-008).

실험을 돌리지 않는다. 기준을 사후에 조정하지 않는다.
"""

from lsrr.gates.criteria import (
    GateResult,
    gate_anytime_increasing,
    gate_beats_baseline,
    gate_delta_decreasing,
    gate_no_collapse,
)
from lsrr.gates.definitions import PHASE_0, PHASES, Phase0Thresholds
from lsrr.gates.report import PhaseReport, build_report

__all__ = (
    "GateResult",
    "gate_no_collapse",
    "gate_anytime_increasing",
    "gate_beats_baseline",
    "gate_delta_decreasing",
    "PHASE_0",
    "PHASES",
    "Phase0Thresholds",
    "PhaseReport",
    "build_report",
)
