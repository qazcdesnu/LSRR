"""절단 BPTT 윈도 (제안서 §5).

    "학습 시 M을 로그정규 분포에서 샘플링하고 마지막 k 사이클만 역전파한다."

**윈도 경계를 공개 API로 노출한다.** 깊은 감독은 윈도 안에서만 샘플링해야
하기 때문이다 — 윈도 밖 상태는 detach되어 있어 감독을 걸어도 엔진에
그래디언트가 가지 않고, 계산만 낭비하며 손실 스케일만 흔든다 (ADR-006).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TBPTTWindow:
    """[start, M) 구간의 사이클만 그래디언트를 흘린다.

    Attributes:
        start: 그래디언트가 흐르기 시작하는 사이클 인덱스 (0-indexed).
        M: 전체 사이클 수.
    """

    start: int
    M: int

    @property
    def k(self) -> int:
        return self.M - self.start

    def __contains__(self, m: int) -> bool:
        return self.start <= m < self.M

    def cycles(self) -> list[int]:
        """윈도 안의 사이클 인덱스. 깊은 감독의 샘플링 모집단이다."""
        return list(range(self.start, self.M))

    def should_detach(self, m: int) -> bool:
        """사이클 m 진입 전에 상태를 detach할지."""
        return m < self.start


def make_window(M: int, tbptt_k: int) -> TBPTTWindow:
    """마지막 `tbptt_k` 사이클만 역전파하는 윈도를 만든다.

    `tbptt_k >= M`이면 전 구간이 윈도다(절단 없음).
    """
    if M < 1:
        raise ValueError(f"M은 1 이상이어야 한다: {M}")
    k = max(1, min(int(tbptt_k), M))
    return TBPTTWindow(start=M - k, M=M)


def detach_if_needed(R: torch.Tensor, window: TBPTTWindow, m: int) -> torch.Tensor:
    """윈도 밖이면 그래프를 끊는다."""
    return R.detach() if window.should_detach(m) else R


__all__ = ("TBPTTWindow", "make_window", "detach_if_needed")
