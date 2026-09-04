import pytest
from lsrr.engines.hydra_qs import HydraQSEngine
from lsrr.engines.attn_block import AttentionBlockEngine
from lsrr.utils.flops import count_parameters

def test_param_matching_attn_and_hydra():
    """Test 6: Parameter count difference between AttentionBlock and HydraQS is < 5%."""
    d_model = 512
    n_blocks = 2

    hydra = HydraQSEngine(d_model=d_model, n_blocks=n_blocks)
    attn = AttentionBlockEngine(d_model=d_model, n_blocks=n_blocks, match_hydra_params=True)

    params_hydra = count_parameters(hydra)["total_parameters"]
    params_attn = count_parameters(attn)["total_parameters"]

    diff_rel = abs(params_hydra - params_attn) / params_hydra
    print(f"\n[Test 6] HydraQS params: {params_hydra:,}, AttentionBlock params: {params_attn:,}, Rel diff: {diff_rel*100:.2f}%")

    assert diff_rel < 0.05, f"Parameter difference exceeded 5%! Hydra: {params_hydra}, Attn: {params_attn}, Diff: {diff_rel:.4f}"
    print("[Test 6 Passed] Parameter matching verified under 5% budget difference.")
