import time
from collections import defaultdict
from typing import Dict, Any, List, Optional
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from lsrr.utils.flops import (
    estimate_engine_flops_per_step,
    estimate_backbone_flops_per_forward,
    estimate_decoder_flops_per_token,
    estimate_adapter_flops,
)
from lsrr.interfaces import BaseTerminationRule, BaseDataModule


class MissingTargetError(RuntimeError):
    """Raised when a batch carries no gold answer.

    Scoring against an empty target silently reports 100% on free-form datasets and 0%
    on numeric ones, so this is fatal rather than skipped.
    """


class Evaluator:
    """Evaluates task accuracy, FLOPs, average stopping cycles, and latency.

    Accuracy follows the final-answer exact-match protocol of the latent-reasoning
    baselines: greedy decoding, per-sequence EOS truncation, dataset-specific
    normalization, only the final answer counts.

    Cost accounting includes the single frozen-backbone forward pass. That pass is the
    whole point of the comparison against Coconut-style rollouts, so leaving it out
    would make the reported numbers meaningless.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        dataset_mod: BaseDataModule,
        tokenizer: Any,
        device: torch.device,
        engine_name: str = "hydra_qs",
        d_model: int = 768,
        num_engine_layers: int = 2,
        num_model_layers: int = 12,
        extractor: Optional[Any] = None,
        eval_samples: Optional[List[Any]] = None,
        max_new_tokens: int = 32,
    ):
        self.model = model.to(device)
        self.dataset_mod = dataset_mod
        self.tokenizer = tokenizer
        self.device = device
        self.engine_name = engine_name
        self.d_model = d_model
        self.num_engine_layers = num_engine_layers
        self.num_model_layers = num_model_layers
        self.extractor = extractor
        self.eval_samples = eval_samples
        self.max_new_tokens = max_new_tokens

        # Precompute analytical FLOPs per refinement cycle
        self.flops_per_cycle = estimate_engine_flops_per_step(
            engine_name=engine_name,
            d_model=d_model,
            num_layers=num_engine_layers,
            L=num_model_layers
        )

        self._backbone_flops: Optional[float] = None
        self._backbone_latency_ms: Optional[float] = None

    # ------------------------------------------------------------------ helpers

    def _resolve_target(self, batch: Dict[str, Any], b: int, meta: Dict[str, Any]) -> str:
        """Gold answer for sample b, from cached meta or from the cached target ids."""
        target_str = str(meta.get("answer", "") or "").strip()
        if target_str:
            return target_str

        target_ids = batch.get("target_ids")
        if target_ids is not None:
            lens = batch.get("target_lens")
            t_len = int(lens[b].item()) if lens is not None else target_ids.size(1)
            if t_len > 0:
                return self.tokenizer.decode(target_ids[b, :t_len], skip_special_tokens=True).strip()

        raise MissingTargetError(
            "No gold answer for this sample: cached meta has no 'answer' key and the batch "
            "carries no 'target_ids'. Re-extract the cache (meta now records the answer) or "
            "use a collate_fn that keeps target_ids."
        )

    def _decoder_flops_per_token(self) -> float:
        dec = getattr(self.model, "decoder", None)
        return estimate_decoder_flops_per_token(
            d_model=getattr(dec, "d_model", self.d_model),
            n_layers=len(getattr(getattr(dec, "decoder", None), "layers", [])) or 2,
            vocab_size=getattr(dec, "vocab_size", 50257),
        )

    def measure_backbone_cost(self, batch_size: int = 16, max_batches: int = 8) -> Dict[str, float]:
        """Time and cost one frozen-backbone forward pass per sample.

        Cached H makes the refinement loop look free; this measures the pass that
        produced the cache so the reported totals are honest.
        """
        if self.extractor is None or not self.eval_samples:
            return {}

        questions = [s.question for s in self.eval_samples]
        n_timed, total_tokens, elapsed = 0, 0, 0.0

        for i in range(0, min(len(questions), batch_size * max_batches), batch_size):
            chunk = questions[i:i + batch_size]
            enc = self.extractor.tokenizer(chunk, padding=True, truncation=True, return_tensors="pt")
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad():
                self.extractor.extract_hidden_states(
                    input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]
                )
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            elapsed += time.time() - t0
            n_timed += len(chunk)
            total_tokens += int(enc["attention_mask"].sum().item())

        if n_timed == 0:
            return {}

        avg_seq_len = total_tokens / n_timed
        self._backbone_latency_ms = (elapsed / n_timed) * 1000.0
        self._backbone_flops = estimate_backbone_flops_per_forward(
            num_layers=self.num_model_layers,
            d_model=getattr(self.extractor, "hidden_dim", self.d_model),
            seq_len=int(round(avg_seq_len)),
        )
        return {
            "backbone_latency_ms_per_sample": self._backbone_latency_ms,
            "backbone_flops_per_sample": self._backbone_flops,
            "backbone_avg_question_tokens": avg_seq_len,
            "backbone_timed_samples": n_timed,
        }

    # ------------------------------------------------------------------ evaluation

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
        total_gen_tokens = 0
        per_hop = defaultdict(lambda: [0, 0])  # hops -> [correct, total]

        # Replace model's termination rule temporarily if specified
        orig_rule = getattr(self.model.controller, "termination_rule", None)
        if termination_rule is not None:
            self.model.controller.termination_rule = termination_rule

        eos_id = getattr(self.tokenizer, "eos_token_id", None)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()
        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating"):
                H = batch["H"].to(self.device)
                meta_list = batch.get("meta", [])
                B = H.shape[0]

                # Generate answers (greedy, EOS-truncated per sequence)
                gen_tokens, stopping_cycles, _ = self.model.generate_answer(
                    H, max_new_tokens=self.max_new_tokens
                )
                pred_texts = self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

                if eos_id is not None and gen_tokens.numel() > 0:
                    total_gen_tokens += int((gen_tokens != eos_id).sum().item())
                else:
                    total_gen_tokens += int(gen_tokens.numel())

                for b in range(B):
                    meta = meta_list[b] if b < len(meta_list) else {}
                    target_str = self._resolve_target(batch, b, meta)

                    is_correct = self.dataset_mod.evaluate_answer(pred_texts[b], target_str, meta)
                    correct_count += int(is_correct)
                    total_samples += 1

                    hops = meta.get("hops")
                    if hops is not None:
                        per_hop[int(hops)][0] += int(is_correct)
                        per_hop[int(hops)][1] += 1

                if stopping_cycles is not None:
                    total_cycles += stopping_cycles.sum().item()

                if max_samples and total_samples >= max_samples:
                    break

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_time = time.time() - start_time

        # Restore original rule
        if termination_rule is not None:
            self.model.controller.termination_rule = orig_rule

        acc = correct_count / max(1, total_samples)
        avg_m = total_cycles / max(1, total_samples)
        avg_gen_tokens = total_gen_tokens / max(1, total_samples)
        refine_latency_ms = (elapsed_time / max(1, total_samples)) * 1000.0

        # ---- cost breakdown, backbone included when it has been measured
        engine_flops = avg_m * self.flops_per_cycle
        adapter_flops = estimate_adapter_flops(
            d_in=getattr(self.model, "d_in", self.d_model),
            d_model=self.d_model,
            num_layers=self.num_model_layers,
        )
        decoder_flops = avg_gen_tokens * self._decoder_flops_per_token()

        if self._backbone_flops is None and self.extractor is not None and self.eval_samples:
            self.measure_backbone_cost()
        backbone_flops = self._backbone_flops
        backbone_latency = self._backbone_latency_ms
        has_backbone = backbone_flops is not None

        total_flops = (backbone_flops or 0.0) + adapter_flops + engine_flops + decoder_flops
        total_latency = refine_latency_ms + (backbone_latency or 0.0)

        return {
            "accuracy": acc,
            "correct_count": correct_count,
            "total_samples": total_samples,
            "avg_stopping_cycles": avg_m,
            "avg_generated_tokens": avg_gen_tokens,

            # Latency: the backbone pass is a real, once-per-sample cost.
            "latency_ms_per_sample": total_latency,
            "latency_ms_refinement_only": refine_latency_ms,
            "latency_ms_backbone": backbone_latency,
            "latency_includes_backbone": has_backbone,

            # FLOPs breakdown, so the efficiency claim can be audited term by term.
            "flops_backbone_per_sample": backbone_flops,
            "flops_adapter_per_sample": adapter_flops,
            "flops_engine_per_sample": engine_flops,
            "flops_decoder_per_sample": decoder_flops,
            "flops_total_per_sample": total_flops,
            "flops_includes_backbone": has_backbone,
            # Comparable with the `flops_per_token` figures in configs/published_numbers.yaml.
            "flops_per_answer_token": total_flops / max(1.0, avg_gen_tokens),

            "accuracy_by_hops": {
                str(k): {"accuracy": v[0] / max(1, v[1]), "n": v[1]}
                for k, v in sorted(per_hop.items())
            },
            "total_time_sec": elapsed_time,
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
            rule_instance = TERMINATION_REGISTRY.build(dict(r_cfg))
            res = self.evaluate_rule(loader, termination_rule=rule_instance)
            res["rule_config"] = r_cfg
            pareto_results.append(res)
            print(
                f"[Sweep Rule] {r_cfg} -> Acc: {res['accuracy']:.4f}, "
                f"Avg M: {res['avg_stopping_cycles']:.2f}, "
                f"Latency: {res['latency_ms_per_sample']:.2f}ms "
                f"({'incl.' if res['latency_includes_backbone'] else 'EXCL.'} backbone), "
                f"FLOPs: {res['flops_total_per_sample']:.3e}"
            )
        return pareto_results
