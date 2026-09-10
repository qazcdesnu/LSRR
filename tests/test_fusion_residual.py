"""Verification of the proposal 3.4 representation fusion.

    alpha_l   = softmax(w^T r_l^*)
    h_ssm     = sum_l alpha_l * r_l^*
    h_fusion  = h_orig_L + W_r * h_ssm

with h_orig_L = R0[:, -1, :], the adapter output *before* refinement -- not the
SSM-refined R_star[:, -1, :].
"""
import pytest
import torch
import torch.nn.functional as F

from lsrr.fusion.attention_pooling import AttentionPoolingFusionHead
from lsrr.model import LSRRModel


def _recompute_h_ssm(head: AttentionPoolingFusionHead, R_star: torch.Tensor) -> torch.Tensor:
    scores = head.score_proj(R_star).squeeze(-1)
    alpha = F.softmax(scores, dim=-1)
    return torch.einsum("bl,bld->bd", alpha, R_star)


def _small_model(d_in: int = 32, num_layers: int = 6, d_model: int = None) -> LSRRModel:
    adapter_cfg = {"type": "per_layer_affine+rmsnorm"}
    if d_model is not None:
        adapter_cfg["d_model"] = d_model
    return LSRRModel(
        adapter_cfg=adapter_cfg,
        engine_cfg={"type": "mamba_up", "n_blocks": 1, "d_state": 4},
        fusion_cfg={"type": "attention_pooling", "fusion_type": "residual"},
        decoder_cfg={"type": "trained_light_decoder", "vocab_size": 64, "n_layers": 1, "n_heads": 2},
        iteration_cfg={"train_m": {"type": "fixed", "k": 3}, "tbptt_k": 2},
        termination_cfg={"type": "delta_state", "eps": 1e-3, "m_max": 4},
        d_in=d_in,
        num_layers=num_layers,
    )


def test_fusion_head_satisfies_residual_equation():
    """h_fusion - h_orig_L must equal W_r * h_ssm exactly."""
    torch.manual_seed(0)
    B, L, d = 4, 12, 32
    head = AttentionPoolingFusionHead(d_model=d, d_out=d, fusion_type="residual")

    R_star = torch.randn(B, L, d)
    h_orig_L = torch.randn(B, d)

    h_fusion, alpha = head(R_star, h_orig_L=h_orig_L)

    expected = h_orig_L + head.w_r(_recompute_h_ssm(head, R_star))
    assert torch.allclose(h_fusion, expected, atol=1e-6), "h_fusion != h_orig_L + W_r * h_ssm"

    # alpha is a proper distribution over the L layer slots
    assert alpha.shape == (B, L)
    assert torch.allclose(alpha.sum(dim=-1), torch.ones(B), atol=1e-6)

    # The residual is a genuine bypass: shifting h_orig_L by delta shifts h_fusion by delta.
    delta = torch.randn(B, d)
    h_fusion_shifted, _ = head(R_star, h_orig_L=h_orig_L + delta)
    assert torch.allclose(h_fusion_shifted - h_fusion, delta, atol=1e-6)


def test_fusion_head_rejects_missing_context_residual():
    """The old silent fallback to R_star[:, -1, :] must not come back."""
    head = AttentionPoolingFusionHead(d_model=16, d_out=16, fusion_type="residual")
    with pytest.raises(ValueError, match="h_orig_L is required"):
        head(torch.randn(2, 8, 16))

    with pytest.raises(ValueError, match=r"h_orig_L must be"):
        head(torch.randn(2, 8, 16), h_orig_L=torch.randn(2, 8))


def test_model_fuses_against_r0_not_r_star():
    """LSRRModel must feed the fusion head R0[:, -1, :], not the refined R_star."""
    torch.manual_seed(0)
    model = _small_model()
    model.eval()

    H = torch.randn(3, model.num_layers, model.d_in)
    out = model(H, target_ids=torch.randint(0, 64, (3, 5)), is_eval=False)

    R0, R_star = out["R0"], out["R_star"]

    # Refinement actually moved the state, so the two candidate residuals differ.
    assert not torch.allclose(R0[:, -1, :], R_star[:, -1, :], atol=1e-4), \
        "engine left the state unchanged; test cannot distinguish R0 from R_star"

    assert torch.equal(out["h_orig_L"], R0[:, -1, :])

    h_ssm = _recompute_h_ssm(model.fusion, R_star)
    correct = R0[:, -1, :] + model.fusion.w_r(h_ssm)
    buggy = R_star[:, -1, :] + model.fusion.w_r(h_ssm)

    assert torch.allclose(out["h_fusion"], correct, atol=1e-6)
    assert not torch.allclose(out["h_fusion"], buggy, atol=1e-4)


def test_generate_answer_uses_same_residual():
    """The eval/generation path must fuse against R0 as well."""
    torch.manual_seed(0)
    model = _small_model()
    model.eval()

    H = torch.randn(2, model.num_layers, model.d_in)
    with torch.no_grad():
        R0 = model.adapter(H)
        gen_tokens, stopping_cycles, _ = model.generate_answer(H, max_new_tokens=3)

    assert torch.equal(model.context_residual(R0), R0[:, -1, :])
    assert gen_tokens.shape[0] == 2
    assert stopping_cycles.shape == (2,)


@pytest.mark.parametrize("d_in", [32, 768])
def test_d_model_defaults_to_backbone_hidden_size(d_in):
    """No dimension reduction: every slot runs at the backbone hidden size."""
    model = _small_model(d_in=d_in, num_layers=4)

    assert model.d_model == d_in
    assert model.adapter.d_model == d_in
    assert model.fusion.d_model == d_in
    assert model.fusion.d_out == d_in
    assert model.decoder.d_model == d_in

    H = torch.randn(2, 4, d_in)
    R0 = model.adapter(H)
    assert R0.shape == (2, 4, d_in)
    # h_orig_L keeps the backbone width, so it can be added to W_r * h_ssm unprojected.
    assert model.context_residual(R0).shape == (2, d_in)


def test_explicit_adapter_d_model_still_overrides():
    model = _small_model(d_in=64, num_layers=4, d_model=16)
    assert model.d_model == 16
    assert model.adapter.d_model == 16
    assert model.fusion.d_model == 16
    assert model.decoder.d_model == 16


def test_residual_fusion_rejects_mismatched_output_width():
    with pytest.raises(ValueError, match="must share a dimension"):
        AttentionPoolingFusionHead(d_model=32, d_out=16, fusion_type="residual")
