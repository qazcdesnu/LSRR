import torch
import pytest
from lsrr.engines.hydra_qs import HydraQSCore
from lsrr.engines.mamba_up_down import DirectionalSSMCore

def test_hydra_qs_reduces_to_mamba_up():
    """Test 2: When reverse component is disabled, HydraQS produces numerically identical results to MambaUp."""
    torch.manual_seed(42)
    d_model = 64
    d_state = 8
    d_conv = 3
    expand = 2

    # Instantiate HydraQS with disabled backward branch
    hydra = HydraQSCore(
        d_model=d_model,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        disable_backward=True
    )

    # Instantiate MambaUp
    mamba_up = DirectionalSSMCore(
        d_model=d_model,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        direction="up"
    )

    # Copy shared forward parameters from hydra to mamba_up
    with torch.no_grad():
        mamba_up.in_proj.weight.copy_(hydra.in_proj.weight)
        mamba_up.conv1d.weight.copy_(hydra.conv1d.weight)
        mamba_up.conv1d.bias.copy_(hydra.conv1d.bias)
        mamba_up.x_proj_fwd.weight.copy_(hydra.x_proj_fwd.weight)
        mamba_up.dt_proj_fwd.weight.copy_(hydra.dt_proj_fwd.weight)
        mamba_up.dt_proj_fwd.bias.copy_(hydra.dt_proj_fwd.bias)
        mamba_up.A_log_fwd.copy_(hydra.A_log_fwd)
        mamba_up.D.copy_(hydra.D)
        mamba_up.out_proj.weight.copy_(hydra.out_proj.weight)

    hydra.eval()
    mamba_up.eval()

    # Create random test input
    x = torch.randn(2, 12, d_model)

    with torch.no_grad():
        out_hydra = hydra(x)
        out_mamba = mamba_up(x)

    diff = (out_hydra - out_mamba).abs().max().item()
    print(f"\n[Test 2 Passed] Max absolute difference: {diff:.2e}")
    assert diff < 1e-6, f"HydraQS and MambaUp outputs diverged! Max diff: {diff}"
