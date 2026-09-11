"""Hydra 이식 충실도 — M4 완료 판정과 레거시 폐기 판정의 근거.

이 파일이 통과하면 레거시 `runs/`·`caches/`의 대조 가치가 소멸한다
(`ROADMAP.md` «Legacy_LSRR 폐기 계획»).

검증 대상은 quasiseparable 항등식이다.

    Y = shift(SS_fwd(X)) + flip(shift(SS_bwd(flip(X)))) + D·X
                        — Hwang, Lahoti, Dao, Gu; arXiv:2407.09941

레거시에서 이 shift가 누락된 채로 있었고, 그 상태에서는 `hydra_qs`가
`bidir_add`와 **비트 단위로 동일**해 Ablation C의 핵심 비교가 무의미했다.
아래 `test_hydra_qs_is_not_bidir_add`가 그 회귀를 고정한다.

이식: Legacy_LSRR/tests/{test_hydra_quasiseparable,test_engine_equiv,test_param_matching}.py
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from lsrr.core.errors import ConfigError
from lsrr.core.registry import ENGINE_REGISTRY
from lsrr.engine.budget import (
    count_core_params,
    solve_attention_d_ffn,
    target_core_params,
)
from lsrr.engine.core_attention import AttentionCore
from lsrr.engine.core_hydra import HydraQSCore
from lsrr.engine.core_mamba import DirectionalSSMCore
from lsrr.engine.scan import selective_scan_sequential, shift_layers

D_MODEL = 32
D_STATE = 4


def _copy_shared(dst: torch.nn.Module, src: torch.nn.Module, directions: tuple[str, ...]) -> None:
    """두 코어의 공유 파라미터를 맞춘다. 구조 차이만 남기기 위함이다."""
    with torch.no_grad():
        dst.in_proj.weight.copy_(src.in_proj.weight)
        dst.conv1d.weight.copy_(src.conv1d.weight)
        dst.conv1d.bias.copy_(src.conv1d.bias)
        for d in directions:
            getattr(dst, f"x_proj_{d}").weight.copy_(getattr(src, f"x_proj_{d}").weight)
            getattr(dst, f"dt_proj_{d}").weight.copy_(getattr(src, f"dt_proj_{d}").weight)
            getattr(dst, f"dt_proj_{d}").bias.copy_(getattr(src, f"dt_proj_{d}").bias)
            getattr(dst, f"A_log_{d}").copy_(getattr(src, f"A_log_{d}"))
        dst.D.copy_(src.D)
        dst.out_proj.weight.copy_(src.out_proj.weight)


def _toy_scan(u: torch.Tensor) -> torch.Tensor:
    B, L, D = u.shape
    return selective_scan_sequential(
        u,
        torch.full_like(u, 0.5),
        -torch.ones(D, D_STATE),
        torch.ones(B, L, D_STATE) * 0.5,
        torch.ones(B, L, D_STATE) * 0.5,
    )


def _self_dependence(y: torch.Tensor, u: torch.Tensor, l: int) -> float:
    """∂y[l]/∂u[l] 의 크기 — 위치 l 이 자기 자신에게 기여하는 정도."""
    g = torch.autograd.grad(y[0, l].sum(), u, retain_graph=True)[0]
    return float(g[0, l].abs().sum())


# ---------------------------------------------------------------- shift 자체

def test_shift_matches_reference_roll():
    """참조 구현의 roll-by-one + 0번 위치 영치와 동일해야 한다."""
    y = torch.randn(2, 6, 4)
    rolled = torch.roll(y, shifts=1, dims=1)
    rolled[:, 0, :] = 0.0
    assert torch.equal(shift_layers(y), rolled)


def test_shift_removes_own_position_term():
    """shift 후에는 위치 l 의 스캔 출력이 u[l] 에 의존하지 않는다."""
    torch.manual_seed(0)
    u = torch.randn(1, 8, 4, requires_grad=True)
    assert _self_dependence(_toy_scan(u), u, 4) > 0.0
    assert _self_dependence(shift_layers(_toy_scan(u)), u, 4) == pytest.approx(0.0)


def test_diagonal_comes_only_from_D():
    """quasiseparable 구성에서 대각 기여는 D 스킵 하나뿐이어야 한다."""
    torch.manual_seed(0)
    u = torch.randn(1, 8, 4, requires_grad=True)
    flip = lambda t: torch.flip(t, dims=[1])  # noqa: E731

    quasisep = shift_layers(_toy_scan(u)) + flip(shift_layers(_toy_scan(flip(u)))) + u
    naive = _toy_scan(u) + flip(_toy_scan(flip(u))) + u

    d_only = _self_dependence(u * 1.0, u, 4)  # D 스킵 항만의 기여
    assert _self_dependence(quasisep, u, 4) == pytest.approx(d_only)
    # shift 가 없으면 대각이 세 번 계산된다.
    assert _self_dependence(naive, u, 4) > d_only


# ---------------------------------------------------------------- 코어 항등식

def test_forward_matches_written_equation():
    """블록 전체가 docstring 의 수식과 일치한다."""
    torch.manual_seed(3)
    core = HydraQSCore(d_model=D_MODEL, d_state=D_STATE, d_conv=3, expand=2).eval()
    x = torch.randn(2, 10, D_MODEL)

    with torch.no_grad():
        out = core(x)
        u, z = torch.split(core.in_proj(x), [core.d_inner, core.d_inner], dim=-1)
        L = x.size(1)
        u_act = F.silu(core.conv1d(u.transpose(1, 2))[:, :, :L].transpose(1, 2))
        fwd = core._run_branch(u_act, core.x_proj_fwd, core.dt_proj_fwd, core.A_log_fwd)
        bwd = core._run_branch(
            torch.flip(u_act, [1]), core.x_proj_bwd, core.dt_proj_bwd, core.A_log_bwd
        )
        y = (
            shift_layers(fwd)
            + torch.flip(shift_layers(bwd), [1])
            + core.D * u_act
        )
        expected = core.out_proj(y * F.silu(z))

    assert torch.allclose(out, expected, atol=1e-6)


def test_hydra_qs_is_not_bidir_add():
    """Ablation C 의 핵심 비교가 성립하는가 — shift 누락 회귀 고정.

    레거시에서 이 둘의 출력 차이가 정확히 0이었다.
    """
    torch.manual_seed(0)
    hydra = HydraQSCore(d_model=D_MODEL, d_state=D_STATE, d_conv=3, expand=2).eval()
    bidir = DirectionalSSMCore(
        d_model=D_MODEL, d_state=D_STATE, d_conv=3, expand=2, direction="bidir_add"
    ).eval()
    _copy_shared(bidir, hydra, ("fwd", "bwd"))

    x = torch.randn(2, 12, D_MODEL)
    with torch.no_grad():
        diff = (hydra(x) - bidir(x)).abs().max().item()
    assert diff > 1e-4, "hydra_qs 가 bidir_add 로 축퇴했다 — shift 가 빠졌다"


def test_unidirectional_hydra_equals_mamba_up():
    """스캔이 하나면 중복 계산이 없으므로 shift 없이 semiseparable 로 축퇴한다."""
    torch.manual_seed(42)
    hydra = HydraQSCore(
        d_model=D_MODEL, d_state=D_STATE, d_conv=3, expand=2, disable_backward=True
    ).eval()
    mamba = DirectionalSSMCore(
        d_model=D_MODEL, d_state=D_STATE, d_conv=3, expand=2, direction="up"
    ).eval()
    _copy_shared(mamba, hydra, ("fwd",))

    x = torch.randn(2, 12, D_MODEL)
    with torch.no_grad():
        assert (hydra(x) - mamba(x)).abs().max().item() < 1e-6


# ---------------------------------------------------------------- 예산 정합

@pytest.mark.parametrize("d_model", [64, 256])
def test_attention_budget_matches_hydra(d_model):
    """Ablation C 의 공정성 — 어텐션 코어를 기준 엔진 예산에 맞춘다 (ADR-009)."""
    target = target_core_params("hydra_qs", d_model=d_model, d_state=D_STATE, expand=2)
    d_ffn = solve_attention_d_ffn(target, d_model)
    actual = count_core_params(AttentionCore(d_model=d_model, d_ffn=d_ffn))
    assert abs(actual - target) / target < 0.05


def test_budget_mismatch_is_fatal():
    """정합 실패는 경고가 아니라 예외다 (규약 §3)."""
    from lsrr.engine.budget import assert_budget_match

    core = AttentionCore(d_model=64, d_ffn=8)
    with pytest.raises(ConfigError, match="예산 정합 실패"):
        assert_budget_match(core, target_params=10_000_000, engine_type="attn_block",
                            reference="hydra_qs")


def test_unsupported_budget_reference_is_rejected():
    with pytest.raises(ConfigError, match="기준 엔진으로 쓸 수 없다"):
        target_core_params("mlp_onepass", d_model=64)


# ---------------------------------------------------------------- 레지스트리 계약

@pytest.mark.parametrize(
    "key", ["hydra_qs", "mamba_up", "mamba_down", "bidir_add", "attn_block", "mlp_onepass"]
)
def test_every_ablation_c_engine_builds_and_preserves_shape(key):
    """Ablation C 스윕의 모든 원소가 같은 인터페이스로 조립된다."""
    engine = ENGINE_REGISTRY.build(
        {"type": key},
        d_model=D_MODEL,
        d_state=D_STATE,
        n_blocks=1,
        expand=2,
        max_cycles=8,
    )
    R0 = torch.randn(2, 12, D_MODEL)
    R1 = engine.forward_step(R0, R0, 0)
    assert R1.shape == R0.shape
    assert torch.isfinite(R1).all()


def test_engine_does_not_own_the_loop():
    """엔진은 `m` 을 조건 입력으로만 받는다 (제안서 §4.2, 인터페이스 계약)."""
    engine = ENGINE_REGISTRY.build(
        {"type": "hydra_qs"}, d_model=D_MODEL, d_state=D_STATE, n_blocks=1, max_cycles=8
    )
    R0 = torch.randn(2, 12, D_MODEL)
    torch.manual_seed(0)
    a = engine.forward_step(R0, R0, 0)
    torch.manual_seed(0)
    b = engine.forward_step(R0, R0, 3)
    # m 이 다르면 사이클 임베딩 때문에 결과가 달라야 한다 — 내부 상태가 아니라
    # 인자로만 사이클을 안다는 증거다.
    assert not torch.allclose(a, b, atol=1e-6)
