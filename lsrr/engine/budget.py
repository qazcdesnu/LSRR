"""파라미터 예산 정합 — Ablation C의 공정성 장치 (ADR-009, 제안서 §7).

Ablation C는 **믹서 행렬 클래스**를 비교한다: 대각(MLP) / dense(어텐션) /
semiseparable(단방향 SSM) / quasiseparable(Hydra). 파라미터 수가 다르면 "구조가
좋아서"와 "파라미터가 많아서"를 구분할 수 없으므로, 비교 대상들의 예산을 기준
엔진에 맞춘다.

정합은 조립 시점에 수행한다 — `config/validate.py`는 설정 형태만 보고, 실제
파라미터 수는 코어를 만들어 봐야 알 수 있기 때문이다.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from lsrr.core.errors import ConfigError

#: 허용 상대 오차. 이 값을 넘으면 조립을 실패시킨다.
BUDGET_TOLERANCE = 0.05


def count_core_params(core: nn.Module) -> int:
    """코어 1블록의 파라미터 수."""
    return sum(p.numel() for p in core.parameters())


def target_core_params(engine_type: str, d_model: int, **kwargs: Any) -> int:
    """기준 엔진 코어 1블록의 파라미터 수를 센다.

    코어만 세고 래퍼는 제외한다 — 래퍼(감쇠·재주입·사이클 임베딩)는 모든 엔진이
    공유하므로 비교에서 상쇄되고, 여기 포함하면 Ablation C와 D가 교락된다.
    """
    from lsrr.engine.core_hydra import HydraQSCore
    from lsrr.engine.core_mamba import DirectionalSSMCore

    if engine_type == "hydra_qs":
        core: nn.Module = HydraQSCore(d_model=d_model, **kwargs)
    elif engine_type in ("mamba_up", "mamba_down", "bidir_add"):
        direction = "up" if engine_type == "mamba_up" else (
            "down" if engine_type == "mamba_down" else "bidir_add"
        )
        core = DirectionalSSMCore(d_model=d_model, direction=direction, **kwargs)
    else:
        raise ConfigError(
            f"engine.budget_match='{engine_type}'는 기준 엔진으로 쓸 수 없다. "
            f"파라미터 수가 d_model에서 결정되는 SSM 계열만 지원한다."
        )
    return count_core_params(core)


def solve_attention_d_ffn(target_params: int, d_model: int, n_heads: int = 8) -> int:
    """목표 예산에 맞는 어텐션 블록의 FFN 폭을 푼다.

    어텐션 블록 파라미터 = QKVO(4·d²+4·d) + LayerNorm 2개(4·d)
                         + FFN(2·d·d_ffn + d_ffn + d)
    이를 `target_params`와 같게 두고 `d_ffn`에 대해 푼다. d_model에서 유도되므로
    백본 폭이 바뀌어도 자동으로 따라간다.
    """
    base = 4 * (d_model**2) + 4 * d_model + 4 * d_model
    d_ffn = (target_params - base - d_model) / (2 * d_model + 1)
    return max(1, int(d_ffn))


def assert_budget_match(
    core: nn.Module,
    target_params: int,
    engine_type: str,
    reference: str,
    tolerance: float = BUDGET_TOLERANCE,
) -> float:
    """예산 정합을 검사하고 상대 오차를 돌려준다. 초과하면 `ConfigError`.

    조용히 넘어가면 Ablation C의 결론이 파라미터 수 차이로 설명될 수 있으므로,
    경고가 아니라 예외다 (규약 §3).
    """
    actual = count_core_params(core)
    if target_params <= 0:
        raise ConfigError(f"기준 엔진 '{reference}'의 파라미터 수가 0이다.")

    rel = abs(actual - target_params) / target_params
    if rel > tolerance:
        raise ConfigError(
            f"예산 정합 실패: engine.type='{engine_type}' 코어가 {actual:,} 파라미터인데 "
            f"기준 '{reference}'는 {target_params:,}이다 (상대 오차 {rel:.1%} > "
            f"{tolerance:.0%}). Ablation C의 비교가 성립하지 않는다."
        )
    return rel


__all__ = (
    "BUDGET_TOLERANCE",
    "count_core_params",
    "target_core_params",
    "solve_attention_d_ffn",
    "assert_budget_match",
)
