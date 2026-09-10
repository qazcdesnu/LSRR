"""손실 타깃 규약 (I6).

제안서 §5: "모든 답 손실(L_NLL, L_DeepSup)의 타깃은 정답 토큰의 one-hot 분포에
대한 표준 cross-entropy(NLL)로 통일한다."

**가장 흔한 사고 지점** (LEGACY_MAP §3):
- `target_ids` — 디코더 **입력**. 패딩이 실제 토큰 id로 채워진다.
- `labels` — 손실 **타깃**. 패딩이 IGNORE_INDEX로 마스킹된다.

둘을 혼동하면 패딩을 학습한다.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from lsrr.core.errors import MissingTargetError
from lsrr.core.invariants import IGNORE_INDEX


def loss_targets(batch: dict[str, Any]) -> torch.Tensor:
    """손실 타깃을 꺼낸다.

    `labels`가 없으면 실패한다 — `target_ids`로 대체하면 패딩을 감독하게 된다.
    """
    labels = batch.get("labels")
    if labels is None:
        raise MissingTargetError(
            "배치에 `labels`가 없다. `target_ids`(디코더 입력)를 손실 타깃으로 "
            "쓰면 패딩을 학습한다 — collate가 labels를 만들어야 한다."
        )
    if int((labels != IGNORE_INDEX).sum()) == 0:
        raise MissingTargetError(
            "배치의 모든 라벨이 마스킹되어 있다(정답 없음). 건너뛰면 자유형 "
            "데이터셋에서 100%, 수치형에서 0%가 조용히 보고된다."
        )
    return labels


def token_nll(
    logits: torch.Tensor,
    labels: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """답 토큰 NLL.

    연속 디코딩의 teacher-forcing은 위치 t의 로짓이 `labels[:, t]`를 예측하도록
    구성되어 있으므로 **별도 shift가 없다** (backbone/continuation.py).

    Args:
        logits: [B, T_a, V]
        labels: [B, T_a] (패딩 = IGNORE_INDEX)
    """
    if logits.shape[:2] != labels.shape:
        raise ValueError(
            f"로짓 {tuple(logits.shape[:2])}과 라벨 {tuple(labels.shape)}의 길이가 "
            f"다르다 — 주입 위치 정렬이 어긋났을 가능성이 크다."
        )
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction=reduction,
    )


def answer_token_count(labels: torch.Tensor) -> int:
    return int((labels != IGNORE_INDEX).sum())


__all__ = ("loss_targets", "token_nll", "answer_token_count", "IGNORE_INDEX")
