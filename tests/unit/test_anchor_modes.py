"""방출 공식의 잔차 앵커 (v2.1 §4.4 개정, ADR-015, F-025·F-026).

    h⁽ᵐ⁾ = s · RMSNorm(W_proj · h_ssm⁽ᵐ⁾)      ← v2 기본 (anchor=none)
    h⁽ᵐ⁾ = h_ctx + W_r · h_ssm⁽ᵐ⁾              ← v1 (anchor=ctx)

이 파일이 고정하는 것:

- **v1 대조군의 보존** — `anchor=ctx` 가 v1 잔차 융합 그대로여야 Ablation A 가
  "방출 구조 비교" 로 남는다. 대조군까지 바꾸면 "융합식 비교" 가 된다.
- **앵커가 궤적 분화를 막는다** — 공통 스탬프가 토큰을 동질화한다 (F-025).
- **램프는 보정 뒤에 걸려야 한다** — 앞에 걸면 RMSNorm 이 σ(g) 를 소거한다.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from lsrr.core.errors import AssemblyError
from lsrr.readout.fusion import AttentionPoolingFusion

B, L, D_MODEL, D_IN = 4, 12, 32, 48


def _head(anchor: str, **kw) -> AttentionPoolingFusion:
    torch.manual_seed(0)
    return AttentionPoolingFusion(
        d_model=D_MODEL, d_out=D_IN, anchor=anchor, **kw
    )


def _states(M: int = 5) -> list[torch.Tensor]:
    torch.manual_seed(1)
    # 사이클마다 확실히 다른 상태 — 엔진이 분화한 상황을 모사한다
    return [torch.randn(B, L, D_MODEL) * (1.0 + 0.5 * m) for m in range(M)]


def _effective_rank(traj: torch.Tensor) -> float:
    ranks = []
    for sample in traj.float():
        sv = torch.linalg.svdvals(sample)
        p = (sv / sv.sum().clamp(min=1e-12)).clamp(min=1e-12)
        ranks.append(float(torch.exp(-(p * p.log()).sum())))
    return sum(ranks) / len(ranks)


def _emit(head: AttentionPoolingFusion, states, h_ctx) -> torch.Tensor:
    with torch.no_grad():
        return torch.stack(
            [head(R, h_ctx, use_anchor=(i == 0))[0] for i, R in enumerate(states)],
            dim=1,
        )


# ---------------------------------------------------------------- 기본값

def test_default_is_no_anchor():
    """v2 기본은 앵커 제거다 (F-026)."""
    assert AttentionPoolingFusion(d_model=D_MODEL, d_out=D_IN).anchor == "none"


def test_unknown_anchor_is_fatal():
    with pytest.raises(AssemblyError, match="anchor"):
        AttentionPoolingFusion(d_model=D_MODEL, d_out=D_IN, anchor="bogus")


# ---------------------------------------------------------------- v1 대조군 보존

def test_ctx_mode_reproduces_the_v1_residual_formula():
    """anchor=ctx 는 h_ctx + W_r·h_ssm 그대로여야 한다 — Ablation A 의 대조군."""
    head = _head("ctx", fusion_type="residual")
    R = torch.randn(B, L, D_MODEL)
    h_ctx = torch.randn(B, D_IN)

    out, alpha = head(R, h_ctx)
    manual = h_ctx + head.w_r(torch.einsum("bl,bld->bd", alpha, R))
    assert torch.allclose(out, manual, atol=1e-6)


def test_none_mode_drops_the_anchor_entirely():
    head = _head("none")
    R = torch.randn(B, L, D_MODEL)
    h_ctx = torch.randn(B, D_IN)

    out, alpha = head(R, h_ctx)
    manual = head.w_r(torch.einsum("bl,bld->bd", alpha, R))
    assert torch.allclose(out, manual, atol=1e-6)
    # h_ctx 를 바꿔도 결과가 변하지 않는다 — 앵커가 정말 빠졌다는 뜻
    other, _ = head(R, torch.randn(B, D_IN) * 100)
    assert torch.allclose(out, other, atol=1e-6)


def test_ctx_mode_does_depend_on_the_anchor():
    """대조군이 실제로 앵커를 쓰는지 — 위 테스트의 짝."""
    head = _head("ctx")
    R = torch.randn(B, L, D_MODEL)
    a, _ = head(R, torch.zeros(B, D_IN))
    b, _ = head(R, torch.randn(B, D_IN) * 100)
    assert not torch.allclose(a, b, atol=1e-3)


# ---------------------------------------------------------------- 분화 (F-025)

def test_anchor_collapses_the_trajectory():
    """공통 스탬프가 토큰을 동질화한다 — 앵커가 클수록 심하다."""
    states = _states()
    h_ctx = torch.randn(B, D_IN) * 200  # 실측의 ‖h_ctx‖≈232 를 모사

    with_anchor = _emit(_head("ctx"), states, h_ctx)
    without = _emit(_head("none"), states, h_ctx)

    cos_a = F.cosine_similarity(with_anchor[:, 0], with_anchor[:, -1], dim=-1).mean()
    cos_n = F.cosine_similarity(without[:, 0], without[:, -1], dim=-1).mean()
    assert float(cos_a) > 0.99, "앵커가 있으면 토큰이 거의 같아야 한다"
    assert float(cos_n) < float(cos_a), "앵커를 빼면 분화한다"
    assert _effective_rank(without) > _effective_rank(with_anchor)


def test_first_mode_anchors_only_the_first_token():
    states = _states(4)
    h_ctx = torch.randn(B, D_IN) * 200
    head = _head("first")
    traj = _emit(head, states, h_ctx)

    # 첫 토큰은 h_ctx 에 의존하고 나머지는 아니다
    traj2 = _emit(head, states, h_ctx * 2)
    assert not torch.allclose(traj[:, 0], traj2[:, 0], atol=1e-3)
    assert torch.allclose(traj[:, 1:], traj2[:, 1:], atol=1e-6)


# ---------------------------------------------------------------- 램프 위치

def test_ramp_scales_the_output():
    head = _head("ramp", ramp_init=-2.0)
    h = torch.randn(B, D_IN)
    gated = head.post_gate(h)
    assert torch.allclose(gated, torch.sigmoid(head.ramp) * h, atol=1e-6)
    assert float(gated.detach().norm()) < float(h.norm())


def test_ramp_before_rms_calibration_would_be_a_no_op():
    """RMSNorm 은 스케일 불변이라 σ(g) 를 정확히 소거한다.

    램프를 보정 **앞**에 걸면 무연산이 된다 — 구현 위치가 의미를 결정한다.
    """
    def rms_calib(x, gain=0.06):
        r = x.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
        return x / r * gain

    x = torch.randn(B, D_IN)
    g = torch.sigmoid(torch.tensor(-2.0))
    assert torch.allclose(rms_calib(x), rms_calib(g * x), atol=1e-6)
    # 뒤에 걸면 실제로 달라진다
    assert not torch.allclose(rms_calib(x), g * rms_calib(x), atol=1e-6)


def test_post_gate_is_identity_for_other_modes():
    for anchor in ("ctx", "none", "first"):
        head = _head(anchor)
        h = torch.randn(B, D_IN)
        assert torch.equal(head.post_gate(h), h)


def test_only_ramp_mode_has_a_ramp_parameter():
    assert hasattr(_head("ramp"), "ramp")
    for anchor in ("ctx", "none", "first"):
        assert not hasattr(_head(anchor), "ramp")
