"""PASS/FAIL 판정문과 킬 스위치 신호 (ADR-008).

판정문은 **근거를 함께 싣는다** — "FAIL"만 있으면 다음에 무엇을 만질지 알 수 없고,
그러면 기준을 낮추고 싶어진다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from lsrr.gates.criteria import GateResult
from lsrr.gates.definitions import PhaseDefinition


@dataclass(frozen=True)
class PhaseReport:
    """한 Phase의 전체 판정."""

    definition: PhaseDefinition
    results: tuple[GateResult, ...]

    @property
    def passed(self) -> bool:
        """모든 게이트가 평가되었고 전부 통과했는가."""
        return all(r.evaluated and r.passed for r in self.results)

    @property
    def kill_switch_triggered(self) -> bool:
        """킬 스위치 게이트가 **측정된 결과로** 실패했는가.

        데이터가 없어 판정하지 못한 것은 킬 스위치가 아니다 — 아키텍처를
        재검토하라는 신호는 실측이 뒷받침해야 한다.
        """
        return any(
            r.is_kill_switch and r.evaluated and not r.passed for r in self.results
        )

    @property
    def incomplete(self) -> tuple[GateResult, ...]:
        """판정 데이터가 없어 평가하지 못한 게이트."""
        return tuple(r for r in self.results if not r.evaluated)

    @property
    def failed(self) -> tuple[GateResult, ...]:
        """측정 결과로 실패한 게이트."""
        return tuple(r for r in self.results if r.evaluated and not r.passed)

    def as_dict(self) -> dict[str, object]:
        return {
            "phase": self.definition.phase,
            "passed": self.passed,
            "kill_switch_triggered": self.kill_switch_triggered,
            "incomplete": [r.gate_id for r in self.incomplete],
            "gates": [
                {
                    "id": r.gate_id,
                    "name": r.name,
                    "status": r.status,
                    "detail": r.detail,
                    "is_kill_switch": r.is_kill_switch,
                    "evaluated": r.evaluated,
                    "evidence": r.evidence,
                }
                for r in self.results
            ],
        }

    def render(self) -> str:
        """사람이 읽는 판정문."""
        lines = [
            f"═══ {self.definition.phase.upper()} 게이트 판정 ═══",
            f"    {self.definition.description}",
            "",
        ]
        for r in self.results:
            mark = "⬜" if not r.evaluated else ("✅" if r.passed else "❌")
            kill = "  ← 킬 스위치" if r.is_kill_switch else ""
            lines.append(f"  {mark} {r.gate_id} {r.name}{kill}")
            lines.append(f"       {r.detail}")
        lines.append("")

        if self.passed:
            lines.append("  판정: PASS — 다음 마일스톤으로 진행한다.")
        elif self.kill_switch_triggered:
            lines.append("  판정: FAIL — 킬 스위치 발동.")
            lines.append("  아래 마일스톤으로 진행하지 않고 아키텍처를 재검토한다.")
            for note in self.definition.notes:
                lines.append(f"    · {note}")
        elif self.failed:
            failed = ", ".join(r.gate_id for r in self.failed)
            lines.append(f"  판정: FAIL — 게이트 {failed} 미통과 (킬 스위치는 아님).")
            lines.append("  해당 게이트의 근거를 먼저 해소한다.")
        else:
            missing = ", ".join(r.gate_id for r in self.incomplete)
            lines.append(f"  판정: 미완 — 게이트 {missing} 의 판정 데이터가 없다.")
            lines.append("  실험을 먼저 돌린다. 데이터 없음은 통과가 아니다.")

        if self.incomplete and (self.kill_switch_triggered or self.failed):
            missing = ", ".join(r.gate_id for r in self.incomplete)
            lines.append(f"  주의: 게이트 {missing} 는 아직 판정되지 않았다.")
        return "\n".join(lines)


def build_report(
    definition: PhaseDefinition, results: Sequence[GateResult]
) -> PhaseReport:
    """게이트 결과를 Phase 정의 순서대로 정렬해 보고서를 만든다.

    정의에 있는 게이트가 결과에 없으면 던진다 — 판정을 빠뜨린 채 PASS가 나오는
    것이 가장 위험한 실패 양식이다.
    """
    by_id = {r.gate_id: r for r in results}
    missing = [g for g in definition.gate_ids if g not in by_id]
    if missing:
        raise ValueError(
            f"게이트 {', '.join(missing)}의 판정 결과가 없다. "
            f"일부만 평가하고 PASS를 내면 게이트가 무의미해진다."
        )
    return PhaseReport(
        definition=definition,
        results=tuple(by_id[g] for g in definition.gate_ids),
    )


__all__ = ("PhaseReport", "build_report")
