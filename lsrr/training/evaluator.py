import time
from typing import Dict, Any, List, Optional
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from lsrr.utils.flops import estimate_engine_flops_per_step
from lsrr.interfaces import BaseTerminationRule, BaseDataModule

class Evaluator:
    """Evaluates task accuracy, FLOPs, average stopping cycles, and latency.
    Supports termination rule sweeps over a single checkpoint to plot Pareto curves.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        dataset_mod: BaseDataModule,
        tokenizer: Any,
        device: torch.device,
        engine_name: str = "hydra_qs",
        d_model: int = 512,
        num_engine_layers: int = 2,
        num_model_layers: int = 12
    ):
        self.model = model.to(device)
        self.dataset_mod = dataset_mod
        self.tokenizer = tokenizer
        self.device = device
        self.engine_name = engine_name
        self.d_model = d_model
        self.num_engine_layers = num_engine_layers
        self.num_model_layers = num_model_layers

        # Precompute analytical FLOPs per refinement cycle
        self.flops_per_cycle = estimate_engine_flops_per_step(
            engine_name=engine_name,
            d_model=d_model,
            num_layers=num_engine_layers,
            L=num_model_layers
        )

    def evaluate_rule(
        self,
        loader: DataLoader,
        termination_rule: Optional[BaseTerminationRule] = None,
        max_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        self.model.eval()
        correct_count = 0
        total_samples = 0
        total_cycles = 0

        # Replace model's termination rule temporarily if specified
        orig_rule = getattr(self.model.controller, "termination_rule", None)
        if termination_rule is not None:
            self.model.controller.termination_rule = termination_rule

        start_time = time.time()
        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating"):
                H = batch["H"].to(self.device)
                meta_list = batch.get("meta", [])
                B = H.shape[0]

                # Generate answers
                gen_tokens, stopping_cycles, _ = self.model.generate_answer(H)

                pred_texts = self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

                for b in range(B):
                    pred_str = pred_texts[b]
                    meta = meta_list[b] if b < len(meta_list) else {}
                    target_str = meta.get("answer", "")
                    if not target_str and "target_ids" in batch:
                        target_str = self.tokenizer.decode(batch["target_ids"][b], skip_special_tokens=True)

                    is_correct = self.dataset_mod.evaluate_answer(pred_str, target_str, meta)
                    if is_correct:
                        correct_count += 1
                    total_samples += 1

                if stopping_cycles is not None:
                    total_cycles += stopping_cycles.sum().item()

                if max_samples and total_samples >= max_samples:
                    break

        elapsed_time = time.time() - start_time

        # Restore original rule
        if termination_rule is not None:
            self.model.controller.termination_rule = orig_rule

        acc = correct_count / max(1, total_samples)
        avg_m = total_cycles / max(1, total_samples)
        latency_ms = (elapsed_time / max(1, total_samples)) * 1000.0
        avg_flops = avg_m * self.flops_per_cycle

        return {
            "accuracy": acc,
            "avg_stopping_cycles": avg_m,
            "latency_ms_per_sample": latency_ms,
            "avg_flops_per_sample": avg_flops,
            "total_samples": total_samples,
            "total_time_sec": elapsed_time
        }

    def sweep_termination_rules(
        self,
        loader: DataLoader,
        rules: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Sweep termination rules on the same checkpoint without retraining."""
        from lsrr.registry import TERMINATION_REGISTRY
        pareto_results = []
        for r_cfg in rules:
            rule_instance = TERMINATION_REGISTRY.build(r_cfg)
            res = self.evaluate_rule(loader, termination_rule=rule_instance)
            res["rule_config"] = r_cfg
            pareto_results.append(res)
            print(f"[Sweep Rule] {r_cfg} -> Acc: {res['accuracy']:.4f}, Avg M: {res['avg_stopping_cycles']:.2f}, Latency: {res['latency_ms_per_sample']:.2f}ms")
        return pareto_results
