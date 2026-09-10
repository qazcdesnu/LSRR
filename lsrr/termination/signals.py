"""종료 신호 계산 (제안서 §4.3).

신호 계산을 규칙에서 분리해 공통화한다 — 규칙마다 같은 계산을 다시 짜면
Ablation B의 비교가 구현 차이에 오염된다.

세 공간을 모두 지원한다: 상태 공간 Δ / 출력 공간 KL / 엔트로피.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def state_delta(R_m: torch.Tensor, R_next: torch.Tensor) -> torch.Tensor:
    """제안서 §4.3의 Δ⁽ᵐ⁾ — 샘플별로 계산한다.

        Δ⁽ᵐ⁾ = (1/L) Σ_l ‖ r_l⁽ᵐ⁺¹⁾ − r_l⁽ᵐ⁾ ‖₂

    Args:
        R_m, R_next: [B, L, d]

    Returns:
        [B] — 배치 평균이 아니라 샘플별 값. 배치 평균으로 정지를 결정하면
        "문제 난이도에 따른 적응적 계산"이라는 주장이 성립하지 않는다.
    """
    return (R_next - R_m).norm(p=2, dim=-1).mean(dim=-1)


def output_kl(
    logits_m: Optional[torch.Tensor], logits_next: Optional[torch.Tensor]
) -> Optional[torch.Tensor]:
    """출력 공간 수렴: KL(P⁽ᵐ⁺¹⁾ ‖ P⁽ᵐ⁾) — 샘플별 [B].

    마지막 위치(첫 답 토큰)의 분포만 본다. 사이클 정제가 답에 미치는 영향이
    가장 먼저 드러나는 지점이기 때문이다.
    """
    if logits_m is None or logits_next is None:
        return None
    log_p = F.log_softmax(logits_m[:, -1, :].float(), dim=-1)
    log_q = F.log_softmax(logits_next[:, -1, :].float(), dim=-1)
    return (log_q.exp() * (log_q - log_p)).sum(dim=-1)


def output_entropy(logits: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """출력 분포의 엔트로피 — 샘플별 [B].

    "확신이 서면 멈춘다"는 종료 규칙의 신호다. Δ·KL과 달리 **변화량이 아니라
    상태**를 보므로, 진동하는 궤적에서도 정지할 수 있다.
    """
    if logits is None:
        return None
    log_p = F.log_softmax(logits[:, -1, :].float(), dim=-1)
    return -(log_p.exp() * log_p).sum(dim=-1)


def relative_delta(
    R_m: torch.Tensor, R_next: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """상태 노름으로 정규화한 Δ — [B].

    절대 Δ의 임계값 ε은 백본·d_model에 따라 스케일이 달라진다. 상대 Δ는
    그 의존을 줄이므로 백본을 바꿔 가며 같은 ε을 쓰고 싶을 때 쓴다.
    """
    scale = R_m.norm(p=2, dim=-1).mean(dim=-1).clamp(min=eps)
    return state_delta(R_m, R_next) / scale


__all__ = ("state_delta", "output_kl", "output_entropy", "relative_delta")
