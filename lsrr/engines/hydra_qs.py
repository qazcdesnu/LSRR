from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from lsrr.interfaces import BaseRefinementEngine
from lsrr.registry import ENGINE_REGISTRY
from lsrr.engines.ssm_core import selective_scan_sequential
from lsrr.engines.wrapper import EngineWrapper

class HydraQSCore(nn.Module):
    """Quasiseparable bidirectional SSM core (Hydra).

        Y = shift(SS_fwd(X)) + flip(shift(SS_bwd(flip(X)))) + D * X

    Follows Hydra (Hwang, Lahoti, Dao, Gu; arXiv:2407.09941), whose reference
    implementation applies the shift as `roll(y, 1, dim=1)` with position 0 zeroed,
    keeps D out of the scans, and adds the skip term exactly once.

    The shift is what makes the mixer quasiseparable: an SSM scan output at position l
    already contains l's own contribution, so summing a forward and a backward scan
    would count the diagonal twice, and the D skip a third time. Shifting removes the
    diagonal from both scans and leaves D as its sole source.

    With `disable_backward=True` there is no second scan and therefore no double count,
    so the shift is not applied and the block reduces to standard (semiseparable) Mamba
    -- which is what makes it numerically identical to MambaUp.

    Deviation from the reference: D here is a static per-channel parameter, where Hydra
    uses a data-dependent per-head projection. This keeps hydra_qs and bidir_add
    differing in exactly one factor (the shift), so ablation C isolates the
    quasiseparable structure rather than confounding it with the skip parameterization.
    """
    def __init__(
        self,
        d_model: int = 768,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        disable_backward: bool = False
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.disable_backward = disable_backward

        # In-projection to (u, z) for gating
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # 1D depthwise conv
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            padding=d_conv - 1,
            groups=self.d_inner
        )

        # Forward scan parameters
        self.x_proj_fwd = nn.Linear(self.d_inner, self.d_inner + 2 * d_state, bias=False)
        self.dt_proj_fwd = nn.Linear(self.d_inner, self.d_inner, bias=True)
        A_init = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log_fwd = nn.Parameter(torch.log(A_init))

        # Backward scan parameters
        self.x_proj_bwd = nn.Linear(self.d_inner, self.d_inner + 2 * d_state, bias=False)
        self.dt_proj_bwd = nn.Linear(self.d_inner, self.d_inner, bias=True)
        self.A_log_bwd = nn.Parameter(torch.log(A_init.clone()))

        # D skip parameter
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Out-projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    @staticmethod
    def _shift(y: torch.Tensor) -> torch.Tensor:
        """Quasiseparable shift: y_shifted[l] = y[l - 1], y_shifted[0] = 0.

        Equivalent to the reference implementation's roll-by-one with position 0 zeroed.
        """
        return torch.cat([torch.zeros_like(y[:, :1]), y[:, :-1]], dim=1)

    def _run_ssm_branch(self, u: torch.Tensor, x_proj: nn.Linear, dt_proj: nn.Linear, A_log: nn.Parameter) -> torch.Tensor:
        B_sz, L, _ = u.shape
        x_dbl = x_proj(u)
        dt_raw, B, C = torch.split(x_dbl, [self.d_inner, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(dt_proj(dt_raw))
        A = -torch.exp(A_log.float())
        y = selective_scan_sequential(u, delta, A, B, C)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: [B, L, d_model]
        Returns:
            out: [B, L, d_model]
        """
        B_sz, L, _ = x.shape
        # Project and split into u and gating z
        xz = self.in_proj(x)
        u, z = torch.split(xz, [self.d_inner, self.d_inner], dim=-1)

        # 1D conv over sequence dimension L
        u_conv = self.conv1d(u.transpose(1, 2))[:, :, :L].transpose(1, 2)
        u_act = F.silu(u_conv)

        # Forward scan: SS_fwd(X)
        y_fwd = self._run_ssm_branch(u_act, self.x_proj_fwd, self.dt_proj_fwd, self.A_log_fwd)

        if not self.disable_backward:
            # Backward scan: flip -> SS_bwd -> flip
            u_flipped = torch.flip(u_act, dims=[1])
            y_bwd_raw = self._run_ssm_branch(u_flipped, self.x_proj_bwd, self.dt_proj_bwd, self.A_log_bwd)

            # Quasiseparable shift, applied in each scan's own direction:
            #   Y = shift(SS_fwd(X)) + flip(shift(SS_bwd(flip(X)))) + D * X
            # Without it the diagonal is counted three times and the mixer is not
            # quasiseparable -- it degenerates into the bidir_add baseline.
            y_fwd = self._shift(y_fwd)
            y_bwd = torch.flip(self._shift(y_bwd_raw), dims=[1])

            y_total = y_fwd + y_bwd + self.D * u_act
        else:
            # Unidirectional: the diagonal lives in the single scan, as in Mamba.
            y_total = y_fwd + self.D * u_act

        # Gated output: y * silu(z)
        y_gated = y_total * F.silu(z)
        out = self.out_proj(y_gated)
        return out

@ENGINE_REGISTRY.register("hydra_qs")
class HydraQSEngine(BaseRefinementEngine):
    def __init__(
        self,
        d_model: int = 768,
        d_state: int = 16,
        n_blocks: int = 2,
        damping_alpha: float = 0.5,
        cycle_embedding: bool = True,
        reinject_r0: str = "gate",
        disable_backward: bool = False,
        **kwargs
    ):
        super().__init__()
        self.d_model = d_model
        self.n_blocks = n_blocks

        layers = []
        for _ in range(n_blocks):
            core = HydraQSCore(
                d_model=d_model,
                d_state=d_state,
                disable_backward=disable_backward
            )
            layers.append(core)
        self.core = nn.Sequential(*layers) if n_blocks > 1 else layers[0]

        self.wrapper = EngineWrapper(
            core_engine=self.core,
            d_model=d_model,
            damping_alpha=damping_alpha,
            cycle_embedding=cycle_embedding,
            reinject_r0=reinject_r0
        )

    def forward_step(self, R_m: torch.Tensor, R0: torch.Tensor, m: int) -> torch.Tensor:
        return self.wrapper(R_m, R0, m)
