"""메모리 파이프라인 — 구성·범위·어댑터 (제안서 §4.1)."""

from __future__ import annotations

import pytest
import torch

from lsrr.core.types import ContextBundle
from lsrr.memory import (
    AddComposer,
    AllLayersScope,
    ConcatComposer,
    FinalOnlyScope,
    GateComposer,
    IdentityAdapter,
    LastOnlyComposer,
    LateBandScope,
    LayerMemoryPipeline,
    MidBandScope,
    PerLayerAffineAdapter,
    SharedAffineAdapter,
)

B, L, D_IN, D_MODEL = 3, 12, 16, 8


def _bundle():
    return ContextBundle(
        H_last=torch.randn(B, L, D_IN),
        h_ctx=torch.randn(B, D_IN),
        H_pool=torch.randn(B, L, D_IN),
    )


# ---------------------------------------------------------------- 범위


def test_band_indices_are_proportional():
    """구간 경계는 레이어 수의 비율이다 — 절대 인덱스면 백본을 바꿀 때 의미가 달라진다."""
    assert MidBandScope(12).layer_indices(12) == [4, 5, 6, 7]
    assert MidBandScope(24).layer_indices(24) == [8, 9, 10, 11, 12, 13, 14, 15]
    assert LateBandScope(12).layer_indices(12) == [8, 9, 10, 11]


def test_final_only_degenerates_layer_axis():
    """Ablation A의 h(L) 단독 조건 — 레이어 축이 1로 축퇴한다."""
    scope = FinalOnlyScope(L)
    out = scope(torch.randn(B, L, D_IN))
    assert out.shape == (B, 1, D_IN)
    assert scope.layer_indices(L) == [L - 1]


def test_final_only_selects_the_last_layer():
    H = torch.randn(B, L, D_IN)
    assert torch.allclose(FinalOnlyScope(L)(H), H[:, -1:, :])


def test_all_layers_is_identity():
    H = torch.randn(B, L, D_IN)
    assert torch.allclose(AllLayersScope(L)(H), H)


def test_band_keeps_at_least_one_layer():
    """비율이 좁아도 빈 축이 나오지 않는다."""
    from lsrr.memory.scoping import _BandScope

    class Narrow(_BandScope):
        band = (0.5, 0.51)

    assert len(Narrow(4).layer_indices(4)) >= 1


def test_invalid_band_rejected():
    from lsrr.memory.scoping import _BandScope

    with pytest.raises(ValueError, match="구간"):
        _BandScope(12, band=(0.7, 0.3))


# ---------------------------------------------------------------- 구성


def test_last_only_ignores_pool():
    """ADR-004의 대조 조건."""
    H_last, H_pool = torch.randn(B, L, D_IN), torch.randn(B, L, D_IN)
    assert torch.allclose(LastOnlyComposer()(H_last, H_pool), H_last)


def test_gate_starts_near_last_only():
    """게이트를 닫은 채 시작해 H_last 단독에서 출발한다."""
    comp = GateComposer(d_in=D_IN, gate_init=-4.0)
    H_last, H_pool = torch.randn(B, L, D_IN), torch.randn(B, L, D_IN)
    out = comp(H_last, H_pool)
    assert (out - H_last).norm() < (H_pool - H_last).norm() * 0.2


def test_gate_records_usage():
    """학습된 게이트 값이 ADR-004의 사후 증거가 된다."""
    comp = GateComposer(d_in=D_IN)
    comp(torch.randn(B, L, D_IN), torch.randn(B, L, D_IN))
    assert comp.last_gate is not None and 0.0 <= float(comp.last_gate) <= 1.0


@pytest.mark.parametrize("cls", [LastOnlyComposer, AddComposer, GateComposer, ConcatComposer])
def test_composers_handle_missing_pool(cls):
    """풀러가 없는 설정에서도 동작해야 한다."""
    comp = cls(d_in=D_IN)
    H_last = torch.randn(B, L, D_IN)
    assert torch.allclose(comp(H_last, None), H_last)


# ---------------------------------------------------------------- 어댑터


def test_per_layer_affine_shapes_and_params():
    ad = PerLayerAffineAdapter(d_in=D_IN, d_model=D_MODEL, num_layers=L)
    assert ad(torch.randn(B, L, D_IN)).shape == (B, L, D_MODEL)
    assert ad.d_model == D_MODEL
    assert ad.weight.shape == (L, D_IN, D_MODEL)


def test_per_layer_affine_uses_different_weights_per_layer():
    """레이어마다 다른 변환이어야 §4.1의 '층간 비정렬' 처방이 성립한다."""
    ad = PerLayerAffineAdapter(d_in=D_IN, d_model=D_MODEL, num_layers=L)
    same = torch.randn(B, 1, D_IN).expand(B, L, D_IN).contiguous()
    out = ad(same)
    assert not torch.allclose(out[:, 0], out[:, 1], atol=1e-4)


def test_adapter_rejects_more_layers_than_built():
    ad = PerLayerAffineAdapter(d_in=D_IN, d_model=D_MODEL, num_layers=4)
    with pytest.raises(ValueError, match="어긋났다"):
        ad(torch.randn(B, 8, D_IN))


def test_shared_affine_uses_same_transform():
    ad = SharedAffineAdapter(
        d_in=D_IN, d_model=D_MODEL, num_layers=L, layer_pos_emb="none"
    )
    same = torch.randn(B, 1, D_IN).expand(B, L, D_IN).contiguous()
    out = ad(same)
    assert torch.allclose(out[:, 0], out[:, 1], atol=1e-5)


def test_identity_adapter_projects_only_when_needed():
    assert isinstance(IdentityAdapter(d_in=D_IN).proj, torch.nn.Identity)
    assert IdentityAdapter(d_in=D_IN, d_model=D_MODEL)(
        torch.randn(B, L, D_IN)
    ).shape == (B, L, D_MODEL)


def test_layer_pos_emb_distinguishes_layers():
    ad = PerLayerAffineAdapter(
        d_in=D_IN, d_model=D_MODEL, num_layers=L, layer_pos_emb="sinusoidal"
    )
    assert ad.layer_emb is not None
    ad_none = PerLayerAffineAdapter(
        d_in=D_IN, d_model=D_MODEL, num_layers=L, layer_pos_emb="none"
    )
    assert ad_none.layer_emb is None


def test_unknown_layer_pos_emb_rejected():
    with pytest.raises(ValueError, match="layer_pos_emb"):
        PerLayerAffineAdapter(d_in=D_IN, d_model=D_MODEL, num_layers=L, layer_pos_emb="bogus")


# ---------------------------------------------------------------- 파이프라인


def test_pipeline_order_is_compose_scope_adapt():
    """scope가 adapt보다 먼저다 — 선택되지 않은 레이어에 파라미터를 주지 않는다."""
    pipe = LayerMemoryPipeline(
        composer=GateComposer(d_in=D_IN),
        scope=MidBandScope(L),
        adapter=PerLayerAffineAdapter(d_in=D_IN, d_model=D_MODEL, num_layers=4),
    )
    out = pipe(_bundle())
    assert out.shape == (B, 4, D_MODEL)


def test_pipeline_exposes_layer_indices():
    """분석에서 층별 귀속을 하려면 원본 인덱스를 알아야 한다."""
    pipe = LayerMemoryPipeline(scope=LateBandScope(L))
    assert pipe.layer_indices(L) == [8, 9, 10, 11]


def test_pipeline_without_slots_is_passthrough():
    pipe = LayerMemoryPipeline()
    bundle = _bundle()
    assert torch.allclose(pipe(bundle), bundle.H_last)


def test_pipeline_diagnostics_report_gate():
    pipe = LayerMemoryPipeline(composer=GateComposer(d_in=D_IN))
    pipe(_bundle())
    assert "composer_gate" in pipe.diagnostics()
