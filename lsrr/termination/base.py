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

    def __init__(self, m_max: int = 32, m_min: int = 1) -> None:
        """
        Args:
            m_max: 폴백 상한. 모든 규칙이 여기서 반드시 정지한다 (I5).
            m_min: 정지 하한. v2.1 §4.3 의 `M = min{m ∈ {M_min..M_max} | Δ<ε}`.
                커리큘럼이 사고 스텝 예산을 강제하는 구간(§5.1 Stage 2)과,
                첫 사이클의 Δ 가 우연히 작아 즉시 멈추는 것을 막는 데 쓴다.
                **M_max 폴백보다 약하다** — 하한과 상한이 충돌하면 상한이 이긴다.
        """
        if int(m_max) < 1:
            raise ValueError(f"m_max는 1 이상이어야 한다: {m_max}")
        if int(m_min) < 1:
            raise ValueError(f"m_min은 1 이상이어야 한다: {m_min}")
        if int(m_min) > int(m_max):
            raise ValueError(
                f"m_min({m_min})이 m_max({m_max})를 넘는다 — 하한을 만족하기 전에 "
                f"폴백이 걸려 의도한 깊이로 돌지 않는다."
            )
        self.m_max = int(m_max)
        self.m_min = int(m_min)
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
            extra={"per_sample_delta": signals.delta_state.detach().cpu().tolist()},
        )

    def should_stop(
        self, signals: TerminationSignals, m: int
    ) -> tuple[torch.Tensor, CycleDiagnostics]:
        if self.stopped is None:
            self.reset(signals.delta_state.shape[0], signals.delta_state.device)

        met = self._criterion(signals, m).to(torch.bool)
        # 하한: M_min 이전에는 규칙이 만족해도 멈추지 않는다 (v2.1 §4.3).
        if (m + 1) < self.m_min:
            met = torch.zeros_like(met)
        fallback = (m + 1) >= self.m_max  # I5 — 하위 클래스도 하한도 우회 못 한다
        self.stopped = self.stopped | met | fallback
        return self.stopped, self._diagnostics(signals, m)

    def needs_logits(self) -> bool:
        return False

    def __repr__(self) -> str:
        return f"{type(self).__name__}(m_min={self.m_min}, m_max={self.m_max})"


__all__ = ("TerminationRuleBase",)
