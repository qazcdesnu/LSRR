"""블록 쌓기 — 증폭이 곱해지지 않아야 한다 (F-029, ADR-017).

선택적 스캔의 순간 항은 `Δ_l·B_l·u_l` 이고 `Δ = softplus(W·u)` 라 세 인자가 모두
입력에 비례한다 — **블록 출력이 입력의 세제곱 규모**다. 블록을 잔차·정규화 없이
이어 붙이면 그 증폭이 곱해져 float32 를 넘치고, 상태 정규화가 `rsqrt(inf)=0` 으로
상태를 0 으로 만든다. 학습이 거기서 죽는다.

여기서 고정하는 것은 "블록을 늘려도 스택이 폭주하지 않는다" 이다. 이것이
성립해야 `n_blocks` 를 Ablation C 의 예산 정합 노브로 쓸 수 있다.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from lsrr.core.registry import ENGINE_REGISTRY
from lsrr.engine.stack import PreNormResidualStack

D = 64
L = 12
B = 4


class _Cubic(nn.Module):
    """입력의 세제곱 규모로 키우는 블록 — 선택적 스캔이 실제로 하는 일이다."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.pow(3)


def test_bare_chaining_explodes_but_the_stack_does_not():
    x = torch.full((B, L, D), 3.0)

    # 이어 붙이면 증폭이 곱해진다: 3 → 3³ → 3⁹.
    bare = nn.Sequential(_Cubic(), _Cubic())
    assert float(bare(x).abs().max()) == pytest.approx(3.0 ** 9)

    # 정규화된 입력을 받으므로 블록마다 O(1) 을 더할 뿐이다.
    stacked = PreNormResidualStack([_Cubic(), _Cubic()])
    assert float(stacked(x).abs().max()) < 10.0


def test_stack_normalizes_the_block_input_not_the_output():
    """잔차 스트림은 살아 있어야 한다 — 출력까지 정규화하면 깊이가 무의미해진다."""
    stack = PreNormResidualStack([nn.Identity()])
    x = torch.randn(B, L, D) * 100.0
    out = stack(x)
    # 입력이 그대로 살아 있고(잔차), 블록에는 정규화된 값이 들어갔다.
    assert torch.allclose(out - x, stack._norm(x), atol=1e-5)


def test_single_block_uses_the_same_structure():
    """`n_blocks` 를 바꿀 때 구조가 달라지면 예산 정합 비교가 블록 수와 교락된다."""
    one = ENGINE_REGISTRY.build({"type": "hydra_qs", "d_model": D, "n_blocks": 1})
    two = ENGINE_REGISTRY.build({"type": "hydra_qs", "d_model": D, "n_blocks": 2})
    assert isinstance(one.core, PreNormResidualStack)
    assert isinstance(two.core, PreNormResidualStack)
    assert len(one.core.blocks) == 1 and len(two.core.blocks) == 2


@pytest.mark.parametrize("engine_type", ["hydra_qs", "mamba_up", "mamba_down", "bidir_add"])
def test_ssm_engines_stay_finite_over_many_cycles(engine_type):
    """M 사이클을 돌려도 상태가 유한해야 한다 — 실측에서 여기가 죽었다."""
    engine = ENGINE_REGISTRY.build(
        {"type": engine_type, "d_model": D, "d_state": 8, "n_blocks": 2}
    )
    R = torch.randn(B, L, D)
    R0 = R.clone()
    with torch.no_grad():
        for m in range(16):
            R = engine.forward_step(R, R0, m)
    assert torch.isfinite(R).all()
    assert float(R.detach().pow(2).mean().sqrt()) == pytest.approx(1.0, abs=0.05)


# ---------------------------------------- 설정 키가 래퍼까지 닿는가 (F-028·F-035)


@pytest.mark.parametrize("engine_type", ["hydra_qs", "mamba_up", "bidir_add", "attn_block", "mlp_onepass"])
def test_engine_config_keys_reach_the_wrapper(engine_type):
    """빌더가 `**_` 로 모르는 키를 삼키면 설정은 그대로인데 다른 실험이 돈다.

    F-035 에서 `engine.mix_norm: true` 가 base.yaml 에 있었는데 조립된 엔진은
    False 였다. `state_norm` 도 같은 경로였다 — 그동안 설정값이 아니라 래퍼
    기본값이 돌았다.
    """
    e = ENGINE_REGISTRY.build({"type": engine_type, "d_model": D, "d_state": 8,
                               "state_norm": "layernorm", "mix_norm": True,
                               "damping_alpha": 0.37})
    assert e.state_norm_type == "layernorm"
    assert e.mix_norm is True
    assert e.damping_alpha == pytest.approx(0.37)
