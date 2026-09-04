import torch
import pytest
from lsrr.termination.rules import DeltaStateTerminationRule, FixedMTerminationRule

def test_delta_state_early_exit_on_converging_sequence():
    """Test 4a: Synthetic converging sequence triggers stop precisely when delta < eps."""
    eps = 0.05
    m_max = 20
    rule = DeltaStateTerminationRule(eps=eps, m_max=m_max)
    rule.reset(batch_size=1, device=torch.device("cpu"))

    B, L, d = 1, 8, 32
    R_current = torch.zeros(B, L, d)

    stopped_at = None
    for m in range(m_max):
        # Step variation decays exponentially: 1 / 2^m
        # Divided by sqrt(d) so the L2 norm over d dimensions equals step_delta exactly
        step_delta = 1.0 / (2.0 ** m)
        R_next = R_current + (step_delta / (d ** 0.5))

        stop_mask, diag = rule.should_stop(R_current, R_next, m=m)
        if stop_mask.item() and stopped_at is None:
            stopped_at = m + 1
            break
        R_current = R_next

    # 1.0 -> 0.5 -> 0.25 -> 0.125 -> 0.0625 -> 0.03125 (< 0.05)
    # m=0: 1.0
    # m=1: 0.5
    # m=2: 0.25
    # m=3: 0.125
    # m=4: 0.0625
    # m=5: 0.03125 -> stops at m=5 (cycle 6)
    assert stopped_at == 6, f"Expected stop at cycle 6, got {stopped_at}"

def test_delta_state_fallback_on_oscillating_sequence():
    """Test 4b: Synthetic non-converging/oscillating sequence falls back to m_max."""
    eps = 0.001
    m_max = 10
    rule = DeltaStateTerminationRule(eps=eps, m_max=m_max)
    rule.reset(batch_size=1, device=torch.device("cpu"))

    B, L, d = 1, 8, 32
    R_current = torch.zeros(B, L, d)

    stopped_at = None
    for m in range(m_max):
        # Oscillate with large amplitude
        sign = 1.0 if (m % 2 == 0) else -1.0
        R_next = R_current + sign * 2.0

        stop_mask, diag = rule.should_stop(R_current, R_next, m=m)
        if stop_mask.item():
            stopped_at = m + 1
            break
        R_current = R_next

    assert stopped_at == m_max, f"Oscillating sequence did not trigger m_max fallback ({m_max}), stopped at {stopped_at}"
    print("\n[Test 4 Passed] Delta state termination and m_max fallback verified.")
