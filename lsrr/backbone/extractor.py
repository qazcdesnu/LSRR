"""레이어별 수직 1열 추출 (제안서 §4.1).

    H = [h⁽¹⁾, h⁽²⁾, …, h⁽ᴸ⁾] ∈ R^{L × d}

질문의 마지막 토큰 위치에서 트랜스포머 1회 순전파로 전 레이어 은닉 상태를
뽑는다. 좌측 패딩을 쓰므로(ADR-011) 모든 샘플의 마지막 실토큰이 인덱스 -1에
놓이고, 인덱싱이 단순해진다.

개작: v1.0:lsrr/backbones/extractor.py
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from lsrr.core.errors import LSRRError


def stack_hidden_states(
    hidden_states: Sequence[torch.Tensor], include_embedding: bool = False
) -> torch.Tensor:
    """HF의 hidden_states 튜플을 레이어 축으로 쌓는다.

    HF는 (L+1)개를 반환한다: index 0은 임베딩 출력, 1..L은 각 블록 출력이며
    마지막 원소는 최종 LayerNorm(ln_f)을 통과한 상태다 — lm_head가 소비하는
    바로 그 벡터다. 레이어 축의 마지막 원소만 정규화 상태가 다른 셈인데,
    이것이 곧 h⁽ᴸ⁾이자 융합 잔차 앵커의 출처이므로 그대로 둔다.

    Args:
        hidden_states: 길이 L+1의 [B, T, d] 텐서 시퀀스.
        include_embedding: 임베딩 출력(index 0)을 레이어 0으로 포함할지.
            L을 바꾸므로 캐시 키·레이어 위치 임베딩 크기에 영향을 준다.

    Returns:
        [B, L, T, d]
    """
    if not hidden_states:
        raise LSRRError("hidden_states가 비어 있다. output_hidden_states=True로 부르라.")
    selected = hidden_states if include_embedding else hidden_states[1:]
    return torch.stack(list(selected), dim=1)


def extract_last_token(
    hidden_stack: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """마지막 실토큰 위치의 레이어별 상태를 뽑는다.

    좌측 패딩이면 모든 샘플의 마지막 실토큰이 -1이므로 단순 슬라이스로 끝난다.
    마스크가 주어지면 마지막 열이 전부 실토큰인지 확인해 우측 패딩을 걸러낸다.

    Args:
        hidden_stack: [B, L, T, d]
        attention_mask: [B, T]

    Returns:
        H_last [B, L, d]
    """
    if attention_mask is not None and not bool(attention_mask[:, -1].all()):
        raise LSRRError(
            "마지막 토큰 위치가 패딩이다 — 우측 패딩된 배치로 보인다. 연속 "
            "디코딩은 좌측 패딩을 요구한다 (ADR-011)."
        )
    return hidden_stack[:, :, -1, :]


def extract_mean(
    hidden_stack: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """마스크 가중 평균 (비교용 position_rule)."""
    if attention_mask is None:
        return hidden_stack.mean(dim=2)
    w = attention_mask[:, None, :, None].to(hidden_stack.dtype)
    return (hidden_stack * w).sum(dim=2) / w.sum(dim=2).clamp(min=1.0)


POSITION_RULES = {"last_token": extract_last_token, "mean": extract_mean}


def extract(
    hidden_stack: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_rule: str = "last_token",
) -> torch.Tensor:
    if position_rule not in POSITION_RULES:
        raise LSRRError(
            f"position_rule '{position_rule}'을 모른다. 가능: {sorted(POSITION_RULES)}"
        )
    return POSITION_RULES[position_rule](hidden_stack, attention_mask)


__all__ = (
    "stack_hidden_states",
    "extract_last_token",
    "extract_mean",
    "extract",
    "POSITION_RULES",
)
