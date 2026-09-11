"""자명해 붕괴 진단 — Phase 0 게이트 ① (제안서 §7).

정제가 "레이어 축을 뭉개서 같은 벡터로 만드는" 자명해로 붕괴하면, 손실은 내려가도
**사고 메모리가 소멸한다**. 그 상태에서 M을 늘려도 정확도가 오를 리 없으므로
게이트 ①은 ②·③보다 먼저 판정해야 하는 위생 검사다.

세 지표를 본다 — 하나만으로는 놓친다.

| 지표 | 붕괴 신호 | 놓치는 경우 |
|---|---|---|
| 유효 랭크 | 1에 가까움 | 레이어별로 크기만 다른 같은 방향 |
| 층간 분산 | 0에 가까움 | 두 개 층만 살아 있고 나머지가 죽은 경우 |
| 쌍별 코사인 유사도 | 1에 가까움 | 부호가 뒤집힌 대칭 붕괴 |

**정규화 후 비교한다.** 유효 랭크와 유사도는 스케일에 불변이어야 하지만, 분산은
`R⁰` 대비 비율로 봐야 의미가 있다 — 절대 분산은 어댑터 초기화에 좌우된다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

#: 유효 랭크가 이 값 아래면 붕괴로 본다 (레이어 축 12 기준).
DEFAULT_MIN_EFFECTIVE_RANK = 2.0
#: 평균 쌍별 유사도가 이 값을 넘으면 붕괴로 본다.
DEFAULT_MAX_SIMILARITY = 0.95
#: R* 층간 분산이 R⁰ 대비 이 비율 아래면 붕괴로 본다.
DEFAULT_MIN_VARIANCE_RATIO = 0.1


@dataclass(frozen=True)
class CollapseReport:
    """붕괴 진단 결과와 그 근거."""

    effective_rank: float
    layer_variance: float
    mean_similarity: float
    variance_ratio: Optional[float] = None

    def is_collapsed(
        self,
        min_rank: float = DEFAULT_MIN_EFFECTIVE_RANK,
        max_similarity: float = DEFAULT_MAX_SIMILARITY,
        min_variance_ratio: float = DEFAULT_MIN_VARIANCE_RATIO,
    ) -> bool:
        """셋 중 **하나라도** 걸리면 붕괴다 — 각 지표가 서로 다른 붕괴를 잡는다."""
        if self.effective_rank < min_rank:
            return True
        if self.mean_similarity > max_similarity:
            return True
        if self.variance_ratio is not None and self.variance_ratio < min_variance_ratio:
            return True
        return False

    def reasons(
        self,
        min_rank: float = DEFAULT_MIN_EFFECTIVE_RANK,
        max_similarity: float = DEFAULT_MAX_SIMILARITY,
        min_variance_ratio: float = DEFAULT_MIN_VARIANCE_RATIO,
    ) -> list[str]:
        """어느 지표가 걸렸는지. 판정문에 그대로 싣는다."""
        out = []
        if self.effective_rank < min_rank:
            out.append(f"유효 랭크 {self.effective_rank:.2f} < {min_rank}")
        if self.mean_similarity > max_similarity:
            out.append(f"평균 쌍별 유사도 {self.mean_similarity:.3f} > {max_similarity}")
        if self.variance_ratio is not None and self.variance_ratio < min_variance_ratio:
            out.append(
                f"층간 분산비 {self.variance_ratio:.3f} < {min_variance_ratio}"
            )
        return out


def effective_rank(R: torch.Tensor) -> float:
    """특이값 분포의 엔트로피 기반 유효 랭크. `[B, L, d] → 스칼라`.

    랭크를 세지 않고 엔트로피를 쓰는 이유는, 수치적 랭크가 임계값에 민감해
    "거의 붕괴"를 놓치기 때문이다. exp(H(p)), p = σ/Σσ 는 연속적으로 변한다.
    """
    x = R.float().reshape(-1, R.shape[-2], R.shape[-1])
    ranks = []
    for sample in x:
        # 중심화하지 않는다. "모든 레이어가 같은 벡터"인 붕괴에서 평균을 빼면
        # 남는 것은 잔여 잡음이고, 그 잡음은 full-rank라 붕괴를 놓친다.
        s = torch.linalg.svdvals(sample)
        total = s.sum()
        if total <= 0:
            ranks.append(1.0)
            continue
        p = s / total
        p = p[p > 0]
        ranks.append(float(torch.exp(-(p * p.log()).sum())))
    return sum(ranks) / len(ranks)


def layer_variance(R: torch.Tensor) -> float:
    """레이어 축을 따라 잰 분산의 평균. `[B, L, d] → 스칼라`."""
    return float(R.float().var(dim=-2, unbiased=False).mean())


def mean_pairwise_similarity(R: torch.Tensor) -> float:
    """서로 다른 레이어 쌍의 평균 코사인 유사도. `[B, L, d] → 스칼라`.

    1에 가까우면 모든 레이어가 같은 방향을 본다 — 레이어 축이 소멸했다.
    """
    x = torch.nn.functional.normalize(R.float(), dim=-1)
    gram = x @ x.transpose(-1, -2)  # [B, L, L]
    L = gram.shape[-1]
    if L < 2:
        return 1.0
    off = ~torch.eye(L, dtype=torch.bool, device=gram.device)
    return float(gram[:, off].mean())


def diagnose(R_star: torch.Tensor, R0: Optional[torch.Tensor] = None) -> CollapseReport:
    """정제 결과의 붕괴 여부를 진단한다.

    Args:
        R_star: `[B, L, d_model]` 최종 사고 메모리.
        R0: `[B, L, d_model]` 초기 메모리. 주면 층간 분산을 비율로 환산한다 —
            절대 분산은 어댑터 초기화에 좌우되어 임계값을 못 정한다.
    """
    if R_star.dim() != 3:
        raise ValueError(f"R_star는 [B, L, d]여야 한다: {tuple(R_star.shape)}")

    var = layer_variance(R_star)
    ratio = None
    if R0 is not None:
        base = layer_variance(R0)
        ratio = (var / base) if base > 0 else None

    return CollapseReport(
        effective_rank=effective_rank(R_star),
        layer_variance=var,
        mean_similarity=mean_pairwise_similarity(R_star),
        variance_ratio=ratio,
    )


__all__ = (
    "CollapseReport",
    "DEFAULT_MAX_SIMILARITY",
    "DEFAULT_MIN_EFFECTIVE_RANK",
    "DEFAULT_MIN_VARIANCE_RATIO",
    "diagnose",
    "effective_rank",
    "layer_variance",
    "mean_pairwise_similarity",
)
