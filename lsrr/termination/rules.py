import math
from typing import Tuple, Optional, Dict, Any, Literal
import torch
import torch.nn.functional as F
from lsrr.interfaces import BaseTerminationRule, StepDiagnostics
from lsrr.registry import TERMINATION_REGISTRY

@TERMINATION_REGISTRY.register("fixed_m")
class FixedMTerminationRule(BaseTerminationRule):
    def __init__(self, m: int = 6, m_max: int = 32, **kwargs):
        self.target_m = m
        self.m_max = m_max
        self.stopped_mask: Optional[torch.Tensor] = None

    def reset(self, batch_size: int, device: torch.device):
        self.stopped_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

    def should_stop(
        self,
        R_m: torch.Tensor,
        R_next: torch.Tensor,
        logits_m: Optional[torch.Tensor] = None,
        logits_next: Optional[torch.Tensor] = None,
        m: int = 0
    ) -> Tuple[torch.Tensor, StepDiagnostics]:
        B = R_m.shape[0]
        if self.stopped_mask is None:
            self.reset(B, R_m.device)

        # Delta state for logging
        delta = (R_next - R_m).norm(dim=-1).mean().item()
        diagnostics = StepDiagnostics(delta_state=delta)

        # Stop when reached target_m or m_max
        should_exit = (m + 1 >= self.target_m) or (m + 1 >= self.m_max)
        if should_exit:
            self.stopped_mask.fill_(True)

        return self.stopped_mask, diagnostics

@TERMINATION_REGISTRY.register("delta_state")
class DeltaStateTerminationRule(BaseTerminationRule):
    """Convergence termination based on state variation:
    Delta^{(m)} = (1/L) * sum_l || r_l^{(m+1)} - r_l^{(m)} ||_2 < eps
    """
    def __init__(
        self,
        eps: float = 1e-3,
        m_max: int = 32,
        norm_type: Literal["l2", "fro"] = "l2",
        **kwargs
    ):
        self.eps = eps
        self.m_max = m_max
        self.norm_type = norm_type
        self.stopped_mask: Optional[torch.Tensor] = None

    def reset(self, batch_size: int, device: torch.device):
        self.stopped_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

    def should_stop(
        self,
        R_m: torch.Tensor,
        R_next: torch.Tensor,
        logits_m: Optional[torch.Tensor] = None,
        logits_next: Optional[torch.Tensor] = None,
        m: int = 0
    ) -> Tuple[torch.Tensor, StepDiagnostics]:
        B, L, _ = R_m.shape
        if self.stopped_mask is None:
            self.reset(B, R_m.device)

        diff = R_next - R_m  # [B, L, d]
        # (1/L) * sum_l || r_l^(m+1) - r_l^(m) ||_2 per sample
        per_sample_delta = diff.norm(p=2, dim=-1).mean(dim=-1)  # [B]
        avg_delta = per_sample_delta.mean().item()

        diagnostics = StepDiagnostics(delta_state=avg_delta)

        # Termination condition: delta < eps OR m+1 >= m_max fallback
        meets_eps = per_sample_delta < self.eps
        fallback = (m + 1 >= self.m_max)

        new_stops = meets_eps | fallback
        self.stopped_mask = self.stopped_mask | new_stops

        return self.stopped_mask, diagnostics

@TERMINATION_REGISTRY.register("kl_output")
class KLOutputTerminationRule(BaseTerminationRule):
    """Output space convergence: KL(P_m || P_{m+1}) < eps."""
    def __init__(self, eps: float = 1e-3, m_max: int = 32, **kwargs):
        self.eps = eps
        self.m_max = m_max
        self.stopped_mask: Optional[torch.Tensor] = None

    def reset(self, batch_size: int, device: torch.device):
        self.stopped_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

    def should_stop(
        self,
        R_m: torch.Tensor,
        R_next: torch.Tensor,
        logits_m: Optional[torch.Tensor] = None,
        logits_next: Optional[torch.Tensor] = None,
        m: int = 0
    ) -> Tuple[torch.Tensor, StepDiagnostics]:
        B = R_m.shape[0]
        if self.stopped_mask is None:
            self.reset(B, R_m.device)

        delta = (R_next - R_m).norm(dim=-1).mean().item()

        if logits_m is not None and logits_next is not None:
            p_m = F.log_softmax(logits_m[:, -1, :], dim=-1)
            p_next = F.softmax(logits_next[:, -1, :], dim=-1)
            # KL divergence: sum p_next * (log p_next - log p_m)
            kl = F.kl_div(p_m, p_next, reduction="batchmean", log_target=False).item()
            meets_eps = (kl < self.eps)
        else:
            kl = None
            meets_eps = False

        diagnostics = StepDiagnostics(delta_state=delta, kl_div=kl)
        fallback = (m + 1 >= self.m_max)
        self.stopped_mask = self.stopped_mask | meets_eps | fallback
        return self.stopped_mask, diagnostics
