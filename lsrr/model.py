from typing import Dict, Any, Optional, Tuple, List
import torch
import torch.nn as nn
from omegaconf import DictConfig

from lsrr.registry import (
    ADAPTER_REGISTRY,
    ENGINE_REGISTRY,
    FUSION_REGISTRY,
    DECODER_REGISTRY,
    TERMINATION_REGISTRY
)
from lsrr.iteration.controller import IterationController
import lsrr.adapters
import lsrr.engines
import lsrr.fusion
import lsrr.decoders
import lsrr.termination


def _resolve_d_model(adapter_cfg: Any, d_in: int) -> int:
    """Resolve the reasoning width d_model.

    Defaults to the backbone hidden size d_in, so no dimension reduction sits between
    the backbone context representation and the reasoning state. This is what lets
    R0[:, -1, :] be handed straight to the fusion residual of proposal 3.4.
    An explicit `adapter.d_model` in the config still wins, which small tests rely on.
    """
    if hasattr(adapter_cfg, "get"):
        explicit = adapter_cfg.get("d_model", None)
        if explicit is not None:
            return int(explicit)
    return int(d_in)


class LSRRModel(nn.Module):
    """Full Layer-State Recurrent Reasoner (LSRR) architecture."""
    def __init__(
        self,
        adapter_cfg: Dict[str, Any],
        engine_cfg: Dict[str, Any],
        fusion_cfg: Dict[str, Any],
        decoder_cfg: Dict[str, Any],
        iteration_cfg: Optional[Dict[str, Any]] = None,
        termination_cfg: Optional[Dict[str, Any]] = None,
        d_in: int = 768,       # default GPT-2 hidden size
        num_layers: int = 12   # default GPT-2 layer count
    ):
        super().__init__()
        # Single source of truth for the reasoning width, shared by every downstream slot.
        d_model = _resolve_d_model(adapter_cfg, d_in)
        self.d_in = d_in
        self.d_model = d_model
        self.num_layers = num_layers

        # Build LayerAdapter
        self.adapter = ADAPTER_REGISTRY.build(
            adapter_cfg,
            d_in=d_in,
            num_layers=num_layers,
            d_model=d_model
        )
        adapter_d_model = int(getattr(self.adapter, "d_model", d_model))
        if adapter_d_model != d_model:
            raise ValueError(
                f"Adapter produced d_model={adapter_d_model} but the model resolved "
                f"d_model={d_model}. Engine, fusion and decoder are all built at the "
                f"resolved width, so the adapter must match it."
            )

        # Build RefinementEngine
        self.engine = ENGINE_REGISTRY.build(
            engine_cfg,
            d_model=d_model
        )

        # Build FusionHead. d_out == d_model keeps h_fusion in the same space as
        # h_orig_L, which the 3.4 residual `h_orig_L + W_r * h_ssm` requires.
        self.fusion = FUSION_REGISTRY.build(
            fusion_cfg,
            d_model=d_model,
            d_out=d_model
        )

        # Build AnswerDecoder
        self.decoder = DECODER_REGISTRY.build(
            decoder_cfg,
            d_model=d_model
        )

        # TerminationRule
        if termination_cfg is not None:
            self.termination_rule = TERMINATION_REGISTRY.build(termination_cfg)
        else:
            self.termination_rule = TERMINATION_REGISTRY.build({"type": "delta_state", "eps": 1e-3, "m_max": 32})

        # IterationController
        train_m_cfg = iteration_cfg.get("train_m", {"type": "fixed", "k": 6}) if iteration_cfg else {"type": "fixed", "k": 6}
        tbptt_k = iteration_cfg.get("tbptt_k", 4) if iteration_cfg else 4
        m_max = termination_cfg.get("m_max", 32) if termination_cfg else 32

        self.controller = IterationController(
            engine=self.engine,
            train_m_cfg=train_m_cfg,
            tbptt_k=tbptt_k,
            termination_rule=self.termination_rule,
            m_max=m_max
        )

    @staticmethod
    def context_residual(R0: torch.Tensor) -> torch.Tensor:
        """Proposal 3.4 residual term h^(L), read off the adapted layer memory.

        R0 is the adapter output, so its last layer slot is the backbone's own top-layer
        context representation *before* any SSM refinement. Using R_star[:, -1, :] here
        instead would make h_fusion a pure function of the SSM output and delete the
        residual bypass the proposal specifies.
        """
        return R0[:, -1, :]

    def forward(
        self,
        H: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
        is_eval: bool = False
    ) -> Dict[str, Any]:
        """Args:
            H: [B, L, d_in] extracted layer hidden states
            target_ids: [B, seq_len] answer token ids for teacher-forcing
            is_eval: whether to use evaluation mode with dynamic termination
        """
        # 1. Adapt H to R0
        R0 = self.adapter(H)  # [B, L, d_model]
        h_orig_L = self.context_residual(R0)  # [B, d_model]

        if not is_eval:
            # Training recursive refinement
            R_star, inter_states, diags = self.controller.run_train(R0)
            stopping_cycles = None
        else:
            # Evaluation with early exit termination
            R_star, diags, stopping_cycles = self.controller.run_eval(
                R0,
                fusion_head=self.fusion,
                decoder=self.decoder,
                h_orig_L=h_orig_L
            )
            inter_states = [R_star]

        # 2. Attention pooling representation fusion with the 3.4 context residual
        h_fusion, alpha_weights = self.fusion(R_star, h_orig_L=h_orig_L)

        # 3. Answer decoding
        logits = self.decoder(h_fusion, target_ids)

        return {
            "logits": logits,
            "R0": R0,
            "R_star": R_star,
            "h_orig_L": h_orig_L,
            "h_fusion": h_fusion,
            "alpha_weights": alpha_weights,
            "intermediate_states": inter_states,
            "diagnostics": diags,
            "stopping_cycles": stopping_cycles,
            "fusion_head": self.fusion,
            "decoder": self.decoder
        }

    def generate_answer(
        self,
        H: torch.Tensor,
        max_new_tokens: int = 32
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Any]]:
        """Evaluation answer generation."""
        R0 = self.adapter(H)
        h_orig_L = self.context_residual(R0)
        R_star, diags, stopping_cycles = self.controller.run_eval(
            R0,
            fusion_head=self.fusion,
            decoder=self.decoder,
            h_orig_L=h_orig_L
        )
        h_fusion, alpha_weights = self.fusion(R_star, h_orig_L=h_orig_L)
        gen_tokens = self.decoder.generate(h_fusion, max_new_tokens=max_new_tokens)
        return gen_tokens, stopping_cycles, diags
