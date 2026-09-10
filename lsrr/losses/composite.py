import random
from typing import Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from lsrr.interfaces import BaseLoss
from lsrr.registry import LOSS_REGISTRY
from lsrr.data.collate import IGNORE_INDEX


def _loss_targets(batch: Dict[str, Any]) -> torch.Tensor:
    """Targets for the answer loss: `labels` when the collate provides them.

    Falling back to `target_ids` keeps ad-hoc callers working, but that tensor pads with
    a real token id, so padding would be supervised.
    """
    labels = batch.get("labels")
    return labels if labels is not None else batch["target_ids"]

@LOSS_REGISTRY.register("answer_nll")
class AnswerNLLLoss(BaseLoss):
    def __init__(self, ignore_index: int = IGNORE_INDEX, **kwargs):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, model_outputs: Dict[str, Any], batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        logits = model_outputs["logits"]  # [B, seq_len, vocab_size]
        # `labels` marks padding with IGNORE_INDEX; `target_ids` pads with a real token id
        # and is the decoder input, so supervising it would train the model on padding.
        targets = _loss_targets(batch)    # [B, seq_len]

        # Shift for next token prediction if target is sequence, or direct classification
        if logits.size(1) == targets.size(1):
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            # Match lengths
            min_len = min(logits.size(1), targets.size(1))
            loss = self.loss_fn(
                logits[:, :min_len].reshape(-1, logits.size(-1)),
                targets[:, :min_len].reshape(-1)
            )
        return {"loss": loss, "answer_nll": loss}

@LOSS_REGISTRY.register("deep_supervision")
class DeepSupervisionLoss(BaseLoss):
    """Supervises intermediate refinement cycles by passing intermediate R through fusion and decoder."""
    def __init__(self, cycles: str = "sample2", **kwargs):
        super().__init__()
        self.cycles = cycles
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    def forward(self, model_outputs: Dict[str, Any], batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        intermediate_states = model_outputs.get("intermediate_states", [])
        if len(intermediate_states) <= 1:
            return {"loss": torch.tensor(0.0, device=model_outputs["logits"].device), "deep_sup": torch.tensor(0.0)}

        fusion_head = model_outputs["fusion_head"]
        decoder = model_outputs["decoder"]
        targets = _loss_targets(batch)
        decoder_inputs = batch.get("target_ids", targets)
        # Intermediate cycles must be fused against the same 3.4 context residual as
        # the final state, otherwise deep supervision optimises a different head.
        h_orig_L = model_outputs.get("h_orig_L")
        if h_orig_L is None:
            h_orig_L = model_outputs["R0"][:, -1, :]

        # Sample intermediate states (excluding 0 and final)
        candidates = intermediate_states[1:-1] if len(intermediate_states) > 2 else intermediate_states[:-1]
        if not candidates:
            candidates = [intermediate_states[0]]

        if self.cycles == "sample2" and len(candidates) >= 2:
            chosen = random.sample(candidates, 2)
        else:
            chosen = [random.choice(candidates)]

        total_loss = 0.0
        for state in chosen:
            h_f, _ = fusion_head(state, h_orig_L=h_orig_L)
            inter_logits = decoder(h_f, decoder_inputs)
            min_len = min(inter_logits.size(1), targets.size(1))
            step_loss = self.loss_fn(
                inter_logits[:, :min_len].reshape(-1, inter_logits.size(-1)),
                targets[:, :min_len].reshape(-1)
            )
            total_loss = total_loss + step_loss

        avg_loss = total_loss / len(chosen)
        return {"loss": avg_loss, "deep_sup": avg_loss}

@LOSS_REGISTRY.register("state_variance_reg")
class StateVarianceRegLoss(BaseLoss):
    """Regularizes state variance across layers to prevent representation collapse or divergence."""
    def __init__(self, target_std: float = 1.0, **kwargs):
        super().__init__()
        self.target_std = target_std

    def forward(self, model_outputs: Dict[str, Any], batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        R_star = model_outputs.get("R_star")  # [B, L, d]
        if R_star is None:
            return {"loss": torch.tensor(0.0), "var_reg": torch.tensor(0.0)}

        # Standard deviation across layers L
        layer_std = R_star.std(dim=1)  # [B, d]
        reg_loss = F.mse_loss(layer_std, torch.full_like(layer_std, self.target_std))
        return {"loss": reg_loss, "var_reg": reg_loss}

class CompositeLoss(nn.Module):
    """Combines a list of loss configurations specified in YAML."""
    def __init__(self, loss_configs: List[Dict[str, Any]]):
        super().__init__()
        self.losses = nn.ModuleList()
        self.weights = []
        self.names = []

        for item in loss_configs:
            cfg = dict(item)
            w = float(cfg.pop("w", 1.0))
            loss_module = LOSS_REGISTRY.build(cfg)
            self.losses.append(loss_module)
            self.weights.append(w)
            self.names.append(cfg.get("type", "loss"))

    def forward(self, model_outputs: Dict[str, Any], batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        total_loss = torch.tensor(0.0, device=model_outputs["logits"].device)
        metrics = {}

        for name, loss_fn, w in zip(self.names, self.losses, self.weights):
            res = loss_fn(model_outputs, batch)
            sub_loss = res["loss"]
            total_loss = total_loss + w * sub_loss
            for k, v in res.items():
                metrics[f"loss_{k}"] = v.detach()

        metrics["total_loss"] = total_loss
        return metrics
