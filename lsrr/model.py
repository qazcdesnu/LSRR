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
        # Build LayerAdapter
        self.adapter = ADAPTER_REGISTRY.build(
            adapter_cfg,
            d_in=d_in,
            num_layers=num_layers
        )
        d_model = getattr(self.adapter, "d_model", 512)

        # Build RefinementEngine
        self.engine = ENGINE_REGISTRY.build(
            engine_cfg,
            d_model=d_model
        )

        # Build FusionHead
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

        if not is_eval:
            # Training recursive refinement
            R_star, inter_states, diags = self.controller.run_train(R0)
            stopping_cycles = None
        else:
            # Evaluation with early exit termination
            R_star, diags, stopping_cycles = self.controller.run_eval(
                R0,
                fusion_head=self.fusion,
                decoder=self.decoder
            )
            inter_states = [R_star]

        # 2. Attention pooling representation fusion
        h_fusion, alpha_weights = self.fusion(R_star)

        # 3. Answer decoding
        logits = self.decoder(h_fusion, target_ids)

        return {
            "logits": logits,
            "R0": R0,
            "R_star": R_star,
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
        R_star, diags, stopping_cycles = self.controller.run_eval(
            R0,
            fusion_head=self.fusion,
            decoder=self.decoder
        )
        h_fusion, alpha_weights = self.fusion(R_star)
        gen_tokens = self.decoder.generate(h_fusion, max_new_tokens=max_new_tokens)
        return gen_tokens, stopping_cycles, diags
