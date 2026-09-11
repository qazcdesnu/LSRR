"""샘플별 조기 종료 래칭 (제안서 §4.3).

종료 판정은 **샘플별**이다 — 배치 평균으로 정지를 결정하면 "문제 난이도에 따른
적응적 계산"이라는 주장이 성립하지 않는다.

이식: v1.0:lsrr/iteration/controller.py:129-141
"""

from __future__ import annotations

import torch


class EarlyExitState:
    """정지한 샘플의 상태를 고정하고 정지 사이클을 기록한다.

    이미 정지한 샘플은 더 갱신하지 않는다 — 갱신하면 "그때 멈췄다"는 기록과
    실제 최종 상태가 어긋나 anytime 곡선과 종료 규칙 분석이 무의미해진다.
    """

    def __init__(self, R0: torch.Tensor, m_max: int) -> None:
        B = R0.shape[0]
        device = R0.device
        self.m_max = m_max
        self.R_final = R0.clone()
        self.finished = torch.zeros(B, dtype=torch.bool, device=device)
        self.stopping_cycles = torch.full((B,), m_max, dtype=torch.long, device=device)

    def update(self, R_next: torch.Tensor, stop_mask: torch.Tensor, m: int) -> None:
        """사이클 m의 결과를 반영한다.

        Args:
            R_next: [B, L, d] 이번 사이클의 갱신 결과.
            stop_mask: [B] 이번 사이클까지 누적된 정지 여부.
            m: 사이클 인덱스 (0-indexed).
        """
        running = ~self.finished
        if running.any():
            self.R_final[running] = R_next[running]

        newly = stop_mask & running
        if newly.any():
            self.stopping_cycles[newly] = m + 1
            self.finished = self.finished | newly

    @property
    def all_done(self) -> bool:
        return bool(self.finished.all())

    @property
    def avg_cycles(self) -> float:
        return float(self.stopping_cycles.float().mean())


__all__ = ("EarlyExitState",)
