"""선택적 스캔 프리미티브 — SSM 계열 코어의 공용 부품.

`core_hydra`와 `core_mamba`가 같은 스캔을 쓴다. 코어끼리 서로를 import하면
Ablation C의 비교 대상들이 서로 의존하게 되므로, 공용 부품은 여기에 둔다.

커스텀 CUDA 커널에 의존하지 않는 순차 구현이다 — 레이어 축 길이는 백본 레이어
수(GPT-2 기준 12)라 짧고, CPU에서 테스트가 돌아야 한다는 규약(§4)을 만족한다.

이식: Legacy_LSRR/lsrr/engines/ssm_core.py
"""

from __future__ import annotations

import torch


def selective_scan_sequential(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
) -> torch.Tensor:
    """선택적 스캔 1방향. `[B, L, D] → [B, L, D]`.

    재귀식은 `x_l = exp(Δ_l A)·x_{l-1} + Δ_l B_l u_l`, 출력은 `y_l = C_l·x_l`.

    D 스킵 항은 **여기에 포함하지 않는다**. quasiseparable 구성에서 D는 대각을
    담당하는 유일한 항이어야 하므로(`core_hydra` 참조), 스캔 밖에서 정확히 한 번
    더해진다. Hydra 참조 구현이 스캔에 `D=None`을 넘기는 것과 같은 이유다.

    Args:
        u: `[B, L, D]` 입력.
        delta: `[B, L, D]` 이산화 간격.
        A: `[D, N]` 상태 전이 (음수여야 안정).
        B: `[B, L, N]` 입력 사영.
        C: `[B, L, N]` 출력 사영.

    Returns:
        `[B, L, D]` 스캔 출력.
    """
    batch, length, dim = u.shape

    delta_a = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # [B, L, D, N]
    delta_b_u = (delta.unsqueeze(-1) * B.unsqueeze(2)) * u.unsqueeze(-1)  # [B, L, D, N]

    state = torch.zeros(batch, dim, A.shape[1], device=u.device, dtype=u.dtype)
    outputs = []
    for l in range(length):
        state = delta_a[:, l] * state + delta_b_u[:, l]  # [B, D, N]
        outputs.append(torch.einsum("bdn,bn->bd", state, C[:, l]))

    return torch.stack(outputs, dim=1)  # [B, L, D]


def shift_layers(y: torch.Tensor) -> torch.Tensor:
    """quasiseparable shift. `y_shifted[l] = y[l-1]`, `y_shifted[0] = 0`.

    Hydra 참조 구현의 `roll(y, 1, dim=1)` + 0번 위치 영치와 동일하다.
    스캔 출력에서 **자기 위치 기여(대각)를 제거**하는 것이 목적이다.
    """
    return torch.cat([torch.zeros_like(y[:, :1]), y[:, :-1]], dim=1)


__all__ = ("selective_scan_sequential", "shift_layers")
