"""Verification that HydraQS implements the quasiseparable mixer of Hydra.

    Y = shift(SS_fwd(X)) + flip(shift(SS_bwd(flip(X)))) + D * X
                                        -- Hwang, Lahoti, Dao, Gu, arXiv:2407.09941

The shift removes each scan's own-position term so the diagonal comes from D alone.
Omitting it counts the diagonal three times and collapses the block into bidir_add.
"""
import torch
import pytest

from lsrr.engines.hydra_qs import HydraQSCore, HydraQSEngine
from lsrr.engines.mamba_up_down import DirectionalSSMCore
from lsrr.engines.ssm_core import selective_scan_sequential


def _copy_shared(dst, src, directions=("fwd",)):
    with torch.no_grad():
        dst.in_proj.weight.copy_(src.in_proj.weight)
        dst.conv1d.weight.copy_(src.conv1d.weight)
        dst.conv1d.bias.copy_(src.conv1d.bias)
        for s in directions:
            getattr(dst, f"x_proj_{s}").weight.copy_(getattr(src, f"x_proj_{s}").weight)
            getattr(dst, f"dt_proj_{s}").weight.copy_(getattr(src, f"dt_proj_{s}").weight)
            getattr(dst, f"dt_proj_{s}").bias.copy_(getattr(src, f"dt_proj_{s}").bias)
            getattr(dst, f"A_log_{s}").copy_(getattr(src, f"A_log_{s}"))
        dst.D.copy_(src.D)
        dst.out_proj.weight.copy_(src.out_proj.weight)


def test_shift_matches_the_reference_roll():
    """The reference applies roll(y, 1, dim=1) with position 0 zeroed."""
    y = torch.randn(2, 6, 4)
    rolled = torch.roll(y, shifts=1, dims=1)
    rolled[:, 0, :] = 0.0
    assert torch.equal(HydraQSCore._shift(y), rolled)


def test_shift_removes_the_own_position_term_from_a_scan():
    """After shifting, position l's scan output no longer depends on u[l]."""
    torch.manual_seed(0)
    B, L, D, N = 1, 8, 4, 3
    u = torch.randn(B, L, D, requires_grad=True)

    def scan(x):
        return selective_scan_sequential(
            x, torch.full_like(x, 0.5), -torch.ones(D, N),
            torch.ones(B, L, N) * 0.5, torch.ones(B, L, N) * 0.5,
        )

    def self_dependence(y, l):
        g = torch.autograd.grad(y[0, l].sum(), u, retain_graph=True)[0]
        return g[0, l].abs().sum().item()

    l = 4
    assert self_dependence(scan(u), l) > 0.0
    assert self_dependence(HydraQSCore._shift(scan(u)), l) == pytest.approx(0.0)


def test_diagonal_is_supplied_only_by_D():
    """The whole block's dependence of y[l] on u[l] must come from the D skip alone."""
    torch.manual_seed(0)
    B, L, D, N = 1, 8, 4, 3
    u = torch.randn(B, L, D, requires_grad=True)

    def scan(x):
        return selective_scan_sequential(
            x, torch.full_like(x, 0.5), -torch.ones(D, N),
            torch.ones(B, L, N) * 0.5, torch.ones(B, L, N) * 0.5,
        )

    shift, flip = HydraQSCore._shift, lambda t: torch.flip(t, dims=[1])
    d_skip = 1.0

    quasisep = shift(scan(u)) + flip(shift(scan(flip(u)))) + d_skip * u
    naive_add = scan(u) + flip(scan(flip(u))) + d_skip * u

    def self_dependence(y, l):
        g = torch.autograd.grad(y[0, l].sum(), u, retain_graph=True)[0]
        return g[0, l].abs().sum().item()

    l = 4
    d_only = self_dependence(d_skip * u, l)
    assert self_dependence(quasisep, l) == pytest.approx(d_only)
    # Without the shift the diagonal is counted three times over.
    assert self_dependence(naive_add, l) > d_only


def test_hydra_qs_is_not_identical_to_bidir_add():
    """Ablation C compares quasiseparable vs naive bidirectional sum.

    Before the shift was applied these two engines were bit-identical, which made the
    comparison vacuous.
    """
    torch.manual_seed(0)
    d = 32
    hydra = HydraQSCore(d_model=d, d_state=8, d_conv=3, expand=2)
    bidir = DirectionalSSMCore(d_model=d, d_state=8, d_conv=3, expand=2, direction="bidir_add")
    _copy_shared(bidir, hydra, directions=("fwd", "bwd"))
    hydra.eval(); bidir.eval()

    x = torch.randn(2, 12, d)
    with torch.no_grad():
        diff = (hydra(x) - bidir(x)).abs().max().item()
    assert diff > 1e-4, "hydra_qs collapsed into the bidir_add baseline"


def test_unidirectional_hydra_still_reduces_to_mamba_up():
    """No second scan means no double count, so no shift -- standard Mamba semantics."""
    torch.manual_seed(42)
    d = 64
    hydra = HydraQSCore(d_model=d, d_state=8, d_conv=3, expand=2, disable_backward=True)
    mamba = DirectionalSSMCore(d_model=d, d_state=8, d_conv=3, expand=2, direction="up")
    _copy_shared(mamba, hydra, directions=("fwd",))
    hydra.eval(); mamba.eval()

    x = torch.randn(2, 12, d)
    with torch.no_grad():
        assert (hydra(x) - mamba(x)).abs().max().item() < 1e-6


def test_forward_matches_the_written_equation():
    """End-to-end: the block equals shift(fwd) + flip(shift(bwd(flip))) + D*u."""
    torch.manual_seed(3)
    import torch.nn.functional as F
    d = 16
    core = HydraQSCore(d_model=d, d_state=4, d_conv=3, expand=2)
    core.eval()
    x = torch.randn(2, 10, d)

    with torch.no_grad():
        out = core(x)

        # Recompute by hand from the module's own pieces.
        xz = core.in_proj(x)
        u, z = torch.split(xz, [core.d_inner, core.d_inner], dim=-1)
        L = x.size(1)
        u_act = F.silu(core.conv1d(u.transpose(1, 2))[:, :, :L].transpose(1, 2))

        fwd = core._run_ssm_branch(u_act, core.x_proj_fwd, core.dt_proj_fwd, core.A_log_fwd)
        bwd = core._run_ssm_branch(torch.flip(u_act, [1]), core.x_proj_bwd, core.dt_proj_bwd, core.A_log_bwd)
        y = core._shift(fwd) + torch.flip(core._shift(bwd), [1]) + core.D * u_act
        expected = core.out_proj(y * F.silu(z))

    assert torch.allclose(out, expected, atol=1e-6)


def test_engine_wrapper_path_still_runs():
    torch.manual_seed(0)
    eng = HydraQSEngine(d_model=32, d_state=4, n_blocks=2)
    R0 = torch.randn(2, 12, 32)
    R1 = eng.forward_step(R0, R0, 0)
    assert R1.shape == R0.shape
    assert torch.isfinite(R1).all()


def test_ssm_core_exposes_only_the_scan_primitive():
    """The dead SSMBlock held a second copy of the shift while the live path had none."""
    import lsrr.engines.ssm_core as m
    assert not hasattr(m, "SSMBlock")
    assert hasattr(m, "selective_scan_sequential")
