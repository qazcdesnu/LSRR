"""블록 쌓기 — pre-norm 잔차 스트림 (FINDINGS F-029).

SSM 계열 코어(`core_hydra`, `core_mamba`)의 블록을 여러 개 쓸 때 필요한 것은
`nn.Sequential` 이 아니다. Mamba·Hydra 참조 구현에서 블록은 **잔차 스트림 안에**
놓이고 진입 전에 정규화된다:

    x ← x + Block(Norm(x))

그냥 이어 붙이면 블록의 증폭이 곱해진다. 선택적 스캔의 순간 항은
`Δ_l · B_l · u_l` 이고 `Δ = softplus(W·u)` 이므로 세 인자가 모두 입력에 비례한다
— **블록 출력이 입력의 세제곱 규모**다. 실측(ProsQA 1 epoch, 2블록): 입력 rms 1
→ 블록0 출력 3.7 → 블록1 출력 2.4×10¹⁰, 그다음 float32 에서 `inf` 로 넘치고
상태 정규화가 `rsqrt(inf) = 0` 으로 상태를 0 으로 만든다. 학습이 거기서 죽는다.

정규화는 블록 입력을, 잔차는 기울기 경로를 지킨다. 둘 다 있어야 `n_blocks` 를
Ablation C 의 예산 정합 노브로 쓸 수 있다 — 블록을 늘리는 것이 발산을 늘리는
일이면 예산 정합 자체가 성립하지 않는다.

`mlp_onepass` 와 `attn_block` 은 이미 각자 잔차를 갖고 있어 이 도우미를 쓰지
않는다. 그래서 여기 정규화는 **게인 없는** 순수 정규화다 — 학습 가능한 이득을
두면 그것이 자라 같은 폭주가 재현된다 (ADR-017).
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class PreNormResidualStack(nn.Module):
    """`x ← x + Block(Norm(x))` 를 블록마다 적용한다.

    블록이 하나뿐이어도 적용한다 — `n_blocks` 를 바꿀 때 구조가 달라지면
    Ablation C 의 예산 정합 비교가 블록 수와 교락된다.
    """

    def __init__(self, blocks: Sequence[nn.Module], eps: float = 1e-6) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.eps = eps

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        f = x.float()
        scale = f.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (f * scale).to(x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = x + block(self._norm(x))
        return x

    def extra_repr(self) -> str:
        return f"n_blocks={len(self.blocks)}"


__all__ = ("PreNormResidualStack",)
