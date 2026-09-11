"""깊은 감독 L_DeepSup — 답 앵커형, 윈도 내 γ 가중 (제안서 §5, ADR-006).

    L_DeepSup = Σ_{m∈S} w_m · NLL(answer | h_fusion⁽ᵐ⁾),   w_m = γ^(M−m)

**왜 답을 앵커로 두는가.** 중간 사이클을 "정답 CoT의 m번째 단계"에 맞추면 잠재
사고를 언어로 되돌리는 셈이고, 언어적 병목을 해소하겠다는 §1의 주장과 어긋난다.
대신 매 사이클이 **같은 최종 답**을 더 잘 맞히도록 압력을 준다 — 궤적의 형태는
엔진이 정하고, 감독은 그 궤적이 답으로 수렴하는지만 본다.

**왜 윈도 안에서만인가.** 윈도 밖 상태는 detach되어 있어 감독을 걸어도 엔진에
그래디언트가 가지 않는다. 계산만 낭비하고 손실 스케일만 흔든다 (ADR-006).

**왜 γ 가중인가.** §5가 지정한 '점진 개선 + 후반 완성' 압력의 형태다. 후반
사이클일수록 가중이 크므로(w_m = γ^(M−m), γ<1) 초반에 답을 확정하라는 압력은
약하고, 마지막에 완성하라는 압력이 강하다. γ=1이면 균등 가중(레거시 거동)이다.

**재검토 조건:** Phase 0 게이트 ②(M 증가 → 정확도 증가)가 실패하면 과잉 감독에
의한 반복 형해화를 의심하고 λ_ds·γ를 먼저 낮춘다 (ADR-006, §5의 명시 처방).

개작: Legacy_LSRR/lsrr/losses/composite.py:DeepSupervisionLoss
      (균등 가중 → γ 가중, 윈도 무관 랜덤 2개 → 윈도 내 샘플링)
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from lsrr.core.interfaces import BaseObjective
from lsrr.core.registry import OBJECTIVE_REGISTRY
from lsrr.core.types import ReadoutResult, ReasoningTrace
from lsrr.objectives.targets import loss_targets, token_nll


@OBJECTIVE_REGISTRY.register("deep_supervision")
class DeepSupervision(BaseObjective):
    """윈도 내 사이클들의 답 NLL을 γ 가중 평균한다.

    Args:
        gamma: 사이클 가중의 감쇠율. `w_m = γ^(M−m)`. 1.0이면 균등 가중.
        num_cycles: 감독할 사이클 수의 상한. 후보가 적으면 그만큼만 쓴다.
        include_final: 마지막 사이클을 후보에 넣을지. 기본은 제외한다 —
            `L_NLL`이 이미 감독하므로 넣으면 같은 항을 두 번 세게 된다.
    """

    def __init__(
        self,
        gamma: float = 0.85,
        num_cycles: int = 2,
        include_final: bool = False,
        **_: Any,
    ) -> None:
        super().__init__()
        if not 0.0 < gamma <= 1.0:
            raise ValueError(f"gamma는 (0, 1] 범위여야 한다: {gamma}")
        self.gamma = float(gamma)
        self.num_cycles = int(num_cycles)
        self.include_final = bool(include_final)

    def _candidates(self, trace: ReasoningTrace) -> list[ReadoutResult]:
        """감독 대상 판독 결과. 윈도 밖과 (기본적으로) 마지막 사이클을 뺀다."""
        start, M = trace.tbptt_window
        last = M - 1
        out = []
        for r in trace.per_cycle_readout:
            if r.m is None or r.logits is None:
                continue
            if not (start <= r.m < M):
                continue
            if not self.include_final and r.m == last:
                continue
            out.append(r)
        return out

    def forward(
        self, trace: ReasoningTrace, batch: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        device = trace.R_star.device if trace.R_star is not None else None
        candidates = self._candidates(trace)

        if not candidates:
            # M=1이거나 윈도에 마지막 사이클밖에 없으면 감독할 것이 없다.
            # 0을 돌려주되 그래프에 연결하지 않는다 — 조용히 건너뛰지 않고
            # 지표로 드러낸다 (규약 §3).
            zero = torch.zeros((), device=device)
            return {
                "loss": zero,
                "deep_sup": zero,
                "deep_sup_cycles": torch.zeros((), device=device),
            }

        # 후보가 상한보다 많으면 뒤쪽(가중이 큰) 사이클을 우선 취한다.
        # 무작위 샘플링은 `recurrence.hooks.sample_supervision_cycles`가 판독
        # 시점에 이미 수행한다 — 여기서 또 뽑으면 이중 샘플링이 된다.
        chosen = sorted(candidates, key=lambda r: r.m)[-self.num_cycles :]

        M = trace.tbptt_window[1]
        targets = loss_targets(batch)

        total: Optional[torch.Tensor] = None
        weight_sum = 0.0
        for r in chosen:
            w = self.gamma ** (M - r.m)
            term = w * token_nll(r.logits, targets)
            total = term if total is None else total + term
            weight_sum += w

        loss = total / weight_sum  # 가중 평균 — γ·M에 따라 스케일이 흔들리지 않게
        return {
            "loss": loss,
            "deep_sup": loss.detach(),
            "deep_sup_cycles": torch.tensor(float(len(chosen)), device=device),
        }


__all__ = ("DeepSupervision",)
