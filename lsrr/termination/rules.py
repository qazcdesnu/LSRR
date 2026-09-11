"""종료 규칙 — Ablation B (제안서 §7).

    "고정 M 스윕 vs Δ<ε(상태 공간) vs KL(출력 공간) vs 엔트로피
     — 수렴 종료의 정확도-연산 파레토 우위 검증"

모든 규칙은 `TerminationRuleBase`를 상속하므로 M_max 폴백을 자동으로 갖는다 (I5).

개작: Legacy_LSRR/lsrr/termination/rules.py
"""

from __future__ import annotations

from typing import Any

import torch

from lsrr.core.errors import ConfigError
from lsrr.core.registry import TERMINATION_REGISTRY
from lsrr.core.types import TerminationSignals
from lsrr.termination.base import TerminationRuleBase


@TERMINATION_REGISTRY.register("fixed_m")
class FixedMRule(TerminationRuleBase):
    """정확히 M회 돌고 멈춘다. Ablation B의 대조 조건이자 M 스윕의 도구."""

    def __init__(self, m: int = 6, m_max: int = 32, m_min: int = 1, **_: Any) -> None:
        super().__init__(m_max=m_max, m_min=m_min)
        if int(m) > self.m_max:
            raise ConfigError(
                f"fixed_m의 m({m})이 m_max({m_max})를 넘는다 — 폴백이 먼저 걸려 "
                f"의도한 깊이로 돌지 않는다."
            )
        self.target_m = int(m)

    def _criterion(self, signals: TerminationSignals, m: int) -> torch.Tensor:
        B = signals.delta_state.shape[0]
        reached = (m + 1) >= self.target_m
        return torch.full(
            (B,), reached, dtype=torch.bool, device=signals.delta_state.device
        )


@TERMINATION_REGISTRY.register("delta_state")
class DeltaStateRule(TerminationRuleBase):
    """상태 공간 수렴: Δ⁽ᵐ⁾ < ε — 제안서 §4.3의 기본.

    Args:
        eps: 임계값.
        relative: True면 상태 노름으로 정규화한 Δ를 쓴다. 백본을 바꿔 가며
            같은 ε을 쓰고 싶을 때 유용하다.
    """

    def __init__(
        self,
        eps: float = 1e-3,
        m_max: int = 32,
        m_min: int = 1,
        relative: bool = False,
        **_: Any,
    ) -> None:
        super().__init__(m_max=m_max, m_min=m_min)
        self.eps = float(eps)
        self.relative = relative

    def _criterion(self, signals: TerminationSignals, m: int) -> torch.Tensor:
        value = signals.extra.get("relative_delta") if self.relative else signals.delta_state
        if value is None:
            value = signals.delta_state
        return value < self.eps


@TERMINATION_REGISTRY.register("kl_output")
class KLOutputRule(TerminationRuleBase):
    """출력 공간 수렴: KL(P⁽ᵐ⁺¹⁾ ‖ P⁽ᵐ⁾) < ε.

    사이클별 로짓을 요구하므로 `recurrence.hooks.readout_per_cycle=true`가
    필요하다 — `config/validate.py`가 이 조합을 로드 시점에 확인한다.
    """

    def __init__(
        self, eps: float = 1e-3, m_max: int = 32, m_min: int = 1, **_: Any
    ) -> None:
        super().__init__(m_max=m_max, m_min=m_min)
        self.eps = float(eps)

    def needs_logits(self) -> bool:
        return True

    def _criterion(self, signals: TerminationSignals, m: int) -> torch.Tensor:
        if signals.kl_div is None:
            # 첫 사이클에는 이전 로짓이 없다. 정지하지 않는다.
            return torch.zeros_like(signals.delta_state, dtype=torch.bool)
        return signals.kl_div < self.eps


@TERMINATION_REGISTRY.register("entropy_output")
class EntropyOutputRule(TerminationRuleBase):
    """출력 엔트로피가 임계 아래로 떨어지면 정지한다.

    Δ·KL과 달리 **변화량이 아니라 상태**를 보므로, 진동하는 궤적에서도 "확신이
    섰으면" 정지할 수 있다. `behavior.py`의 `oscillating` 층에서 다른 규칙과
    어떻게 갈리는지가 Ablation B의 관심사다.
    """

    def __init__(
        self, eps: float = 1.0, m_max: int = 32, m_min: int = 1, **_: Any
    ) -> None:
        super().__init__(m_max=m_max, m_min=m_min)
        self.eps = float(eps)

    def needs_logits(self) -> bool:
        return True

    def _criterion(self, signals: TerminationSignals, m: int) -> torch.Tensor:
        if signals.entropy is None:
            return torch.zeros_like(signals.delta_state, dtype=torch.bool)
        return signals.entropy < self.eps


__all__ = ("FixedMRule", "DeltaStateRule", "KLOutputRule", "EntropyOutputRule")
