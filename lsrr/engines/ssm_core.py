import torch

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
