import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

def selective_scan_sequential(
    u: torch.Tensor,       # [B, L, D]
    delta: torch.Tensor,   # [B, L, D]
    A: torch.Tensor,       # [D, N]
    B: torch.Tensor,       # [B, L, N]
    C: torch.Tensor,       # [B, L, N]
) -> torch.Tensor:
    """Sequential selective scan implementation for SSM.
    Works on CPU and CUDA without custom binary dependency.
    u: [B, L, D]
    delta: [B, L, D]
    A: [D, N]
    B: [B, L, N]
    C: [B, L, N]
    Returns: y: [B, L, D]
    """
    B_sz, L, D = u.shape
    N = A.shape[1]

    # deltaA = exp(einsum(b l d, d n -> b l d n))
    deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # [B, L, D, N]
    # deltaB_u = einsum(b l d, b l n, b l d -> b l d n)
    deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # [B, L, D, N]
    deltaB_u = deltaB * u.unsqueeze(-1)           # [B, L, D, N]

    x = torch.zeros(B_sz, D, N, device=u.device, dtype=u.dtype)
    ys = []
    for l in range(L):
        x = deltaA[:, l] * x + deltaB_u[:, l]  # [B, D, N]
        y_l = torch.einsum("bdn,bn->bd", x, C[:, l])  # [B, D]
        ys.append(y_l)

    return torch.stack(ys, dim=1)  # [B, L, D]

class SSMBlock(nn.Module):
    """Core SSM block (Mamba/SSD style).
    Takes [B, L, d_model], projects to inner dimension, applies depth conv/shift + selective scan + gating.
    """
    def __init__(
        self,
        d_model: int = 512,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        # Projections
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            padding=d_conv - 1,
            groups=self.d_inner
        )

        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, self.d_inner + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

        # Initialize dt_proj
        dt_init_std = 2.0 / math.sqrt(self.d_inner)
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # S4D real initialization for A
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))

        # D skip parameter
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward_scan(
        self,
        u: torch.Tensor,
        reverse: bool = False,
        shift_output: bool = False
    ) -> torch.Tensor:
        """Args:
            u: [B, L, d_inner]
            reverse: whether to scan from top to bottom
            shift_output: whether to shift sequence by 1 (quasiseparable shift)
        Returns:
            y: [B, L, d_inner]
        """
        if reverse:
            u = torch.flip(u, dims=[1])

        B_sz, L, _ = u.shape

        # Parameter derivation from u
        x_dbl = self.x_proj(u)  # [B, L, dt_rank + 2*d_state]
        dt_raw, B, C = torch.split(x_dbl, [self.d_inner, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dt_raw))  # [B, L, d_inner]

        A = -torch.exp(self.A_log.float())  # [d_inner, d_state]

        y = selective_scan_sequential(u, delta, A, B, C)

        if shift_output:
            # Shift by 1 along sequence dimension: y_{l} = y_{l-1}, y_0 = 0
            y = torch.cat([torch.zeros_like(y[:, :1]), y[:, :-1]], dim=1)

        if reverse:
            y = torch.flip(y, dims=[1])

        return y
