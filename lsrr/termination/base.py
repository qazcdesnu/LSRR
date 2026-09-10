"""종료 규칙 기반 클래스 — M_max 폴백을 강제한다 (I5).

제안서 §4.3: "일부 상태가 진동(궤도)하는 사례도 보고되어 있으므로 M_max
폴백을 필수 명세로 둔다."

레거시는 규칙마다 `fallback = (m + 1 >= self.m_max)`를 반복해 적었다 — 새
규칙을 추가할 때 빠뜨리기 쉬운 형태다. 여기서는 하위 클래스가 `_criterion`만
구현하고, 폴백 합성은 기반 클래스가 한다. **우회할 수 없다.**
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Optional

import torch

from lsrr.core.interfaces import BaseTerminationRule
from lsrr.core.types import CycleDiagnostics, TerminationSignals


class TerminationRuleBase(BaseTerminationRule):
    """정지 마스크 누적과 M_max 폴백을 공통 제공한다."""

    def __init__(self, m_max: int = 32) -> None:
        if int(m_max) < 1:
            raise ValueError(f"m_max는 1 이상이어야 한다: {m_max}")
        self.m_max = int(m_max)
        self.stopped: Optional[torch.Tensor] = None

    def reset(self, batch_size: int, device: torch.device) -> None:
        self.stopped = torch.zeros(batch_size, dtype=torch.bool, device=device)

    @abstractmethod
    def _criterion(self, signals: TerminationSignals, m: int) -> torch.Tensor:
        """규칙 고유 판정 → [B] bool. 폴백은 신경 쓰지 않는다."""

    def _diagnostics(self, signals: TerminationSignals, m: int) -> CycleDiagnostics:
        return CycleDiagnostics(
            m=m,
            delta_state=float(signals.delta_state.mean()),
            kl_div=None if signals.kl_div is None else float(signals.kl_div.mean()),
            entropy=None if signals.entropy is None else float(signals.entropy.mean()),
        )

    def should_stop(
        self, signals: TerminationSignals, m: int
    ) -> tuple[torch.Tensor, CycleDiagnostics]:
        if self.stopped is None:
            self.reset(signals.delta_state.shape[0], signals.delta_state.device)

        met = self._criterion(signals, m).to(torch.bool)
        fallback = (m + 1) >= self.m_max  # I5 — 하위 클래스가 우회할 수 없다
        self.stopped = self.stopped | met | fallback
        return self.stopped, self._diagnostics(signals, m)

    def needs_logits(self) -> bool:
        return False

    def __repr__(self) -> str:
        return f"{type(self).__name__}(m_max={self.m_max})"


__all__ = ("TerminationRuleBase",)
