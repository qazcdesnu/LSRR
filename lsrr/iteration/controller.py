import math
import random
from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
from lsrr.interfaces import BaseRefinementEngine, BaseTerminationRule, StepDiagnostics

class IterationController:
    """Manages recursive refinement execution for both training (with TBPTT & M sampling)
    and evaluation (with dynamic termination rules & logging).
    """
    def __init__(
        self,
        engine: BaseRefinementEngine,
        train_m_cfg: Optional[Dict[str, Any]] = None,
        tbptt_k: int = 4,
        termination_rule: Optional[BaseTerminationRule] = None,
        m_max: int = 32
    ):
        self.engine = engine
        self.train_m_cfg = train_m_cfg or {"type": "fixed", "k": 6}
        self.tbptt_k = tbptt_k
        self.termination_rule = termination_rule
        self.m_max = m_max

    def sample_train_m(self) -> int:
        dist_type = self.train_m_cfg.get("type", "fixed")
        if dist_type == "fixed":
            return int(self.train_m_cfg.get("k", 6))
        elif dist_type == "lognormal":
            mean = float(self.train_m_cfg.get("mean", 6.0))
            std = float(self.train_m_cfg.get("std", 1.0))
            max_m = int(self.train_m_cfg.get("max", 16))
            # Log-normal sampling around mean
            mu = math.log(max(1.0, mean))
            val = int(random.lognormvariate(mu, std))
            return max(1, min(val, max_m))
        else:
            return 6

    def run_train(
        self,
        R0: torch.Tensor,
        M: Optional[int] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[StepDiagnostics]]:
        """Training forward pass with truncated BPTT."""
        if M is None:
            M = self.sample_train_m()

        R_current = R0
        intermediate_states = [R_current]
        diagnostics = []

        cutoff_step = max(0, M - self.tbptt_k)

        for m in range(M):
            # Truncated BPTT: detach previous steps before the last tbptt_k cycles
            if m <= cutoff_step and m > 0:
                R_current = R_current.detach()

            R_next = self.engine.forward_step(R_current, R0, m)

            # Record step delta
            with torch.no_grad():
                delta = (R_next - R_current).norm(dim=-1).mean().item()
                diagnostics.append(StepDiagnostics(delta_state=delta))

            R_current = R_next
            intermediate_states.append(R_current)

        return R_current, intermediate_states, diagnostics

    def run_eval(
        self,
        R0: torch.Tensor,
        termination_rule: Optional[BaseTerminationRule] = None,
        fusion_head: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None
    ) -> Tuple[torch.Tensor, List[StepDiagnostics], torch.Tensor]:
        """Evaluation pass with autonomous convergence termination.
        Returns:
            R_star: [B, L, d] final state
            diagnostics: List of StepDiagnostics
            stopping_cycles: Tensor[B] actual cycles taken per sample
        """
        rule = termination_rule or self.termination_rule
        B = R0.shape[0]
        device = R0.device

        if rule is not None:
            rule.reset(B, device)

        R_current = R0
        diagnostics = []
        stopping_cycles = torch.full((B,), self.m_max, dtype=torch.long, device=device)
        is_finished = torch.zeros(B, dtype=torch.bool, device=device)

        # Buffer to keep the final frozen state for samples that stop early
        R_final = R0.clone()

        logits_prev = None

        for m in range(self.m_max):
            with torch.no_grad():
                R_next = self.engine.forward_step(R_current, R0, m)

                # If rule needs output logits
                logits_current = None
                if decoder is not None and fusion_head is not None:
                    h_fusion, _ = fusion_head(R_next)
                    logits_current = decoder(h_fusion)

                if rule is not None:
                    stopped_mask, diag = rule.should_stop(
                        R_current, R_next, logits_prev, logits_current, m
                    )
                else:
                    stopped_mask = torch.zeros(B, dtype=torch.bool, device=device)
                    delta = (R_next - R_current).norm(dim=-1).mean().item()
                    diag = StepDiagnostics(delta_state=delta)

                diagnostics.append(diag)

                # For samples newly stopping at this step, record cycle and latch final state
                newly_stopped = stopped_mask & (~is_finished)
                if newly_stopped.any():
                    stopping_cycles[newly_stopped] = m + 1
                    R_final[newly_stopped] = R_next[newly_stopped]
                    is_finished = is_finished | newly_stopped

                # For running samples, continue updating
                running = ~is_finished
                if running.any():
                    R_final[running] = R_next[running]

                R_current = R_next
                logits_prev = logits_current

                if is_finished.all():
                    break

        return R_final, diagnostics, stopping_cycles
