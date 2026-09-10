#!/usr/bin/env python3
"""Evaluate baseline frozen backbone model directly without Chain-of-Thought (Direct QA).

Evaluates whether the pretrained LM can answer questions zero-shot directly
without any intermediate CoT reasoning steps or latent refinement.

Usage:
    uv run python scripts/eval_backbone_direct.py data=prosqa split=val
    uv run python scripts/eval_backbone_direct.py data=gsm8k split=val
    uv run python scripts/eval_backbone_direct.py backbone=gpt2 data=prosqa split=val num_samples=50
"""

import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add repo root to sys.path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from lsrr.config import load_config_with_cli
from lsrr.registry import DATA_REGISTRY
import lsrr.data

def format_direct_prompt(question: str, prompt_template: Optional[str] = None) -> str:
    """Formats question into a direct prompt without CoT reasoning."""
    q = question.strip()
    if prompt_template:
        return prompt_template.format(question=q)

    # Standard direct answering prompt
    if q.endswith("Answer:") or q.endswith("Answer: "):
        return q
    return f"{q}\nAnswer:"

def evaluate_backbone_direct():
    cfg = load_config_with_cli()

    backbone_cfg = cfg.get("backbone", {"type": "gpt2", "model_name_or_path": "gpt2"})
    model_name = backbone_cfg.get("model_name_or_path", backbone_cfg.get("type", "gpt2"))
    dtype_str = backbone_cfg.get("dtype", "float32")

    data_cfg = cfg.get("data", {"type": "prosqa"})
    split = cfg.get("split", "val")
    batch_size = int(cfg.get("batch_size", 16))
    max_new_tokens = int(cfg.get("max_new_tokens", 16))
    num_samples = cfg.get("num_samples", None)
    prompt_template = cfg.get("prompt_template", None)

    device_req = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_req)

    print(f"[DirectEval] Model: {model_name} on {device}")
    print(f"[DirectEval] Dataset: {data_cfg.get('type')} (split: {split})")
    print(f"[DirectEval] Max new tokens: {max_new_tokens}, Batch size: {batch_size}")

    # 1. Load Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"  # Crucial for batched causal LM generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.bfloat16 if dtype_str == "bfloat16" else (torch.float16 if dtype_str == "float16" else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch_dtype
    ).to(device)
    model.eval()

    # 2. Load Dataset
    dataset_mod = DATA_REGISTRY.build(data_cfg)
    all_samples = dataset_mod.get_split(split)
    if num_samples is not None:
        num_samples = int(num_samples)
        samples = all_samples[:num_samples]
    else:
        samples = all_samples

    total = len(samples)
    print(f"[DirectEval] Evaluating {total} samples (Direct QA, no CoT)...")

    correct = 0
    records = []
    hop_stats: Dict[int, Dict[str, int]] = {}

    start_time = time.time()

    # 3. Batched Inference
    for i in tqdm(range(0, total, batch_size), desc=f"Eval {model_name} Direct"):
        batch_samples = samples[i:i + batch_size]
        batch_prompts = [format_direct_prompt(s.question, prompt_template) for s in batch_samples]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        prompt_len = inputs["input_ids"].shape[1]
        gen_tokens = outputs[:, prompt_len:]
        predictions = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

        for sample, prompt, pred in zip(batch_samples, batch_prompts, predictions):
            cleaned_pred = pred.strip()
            # Also evaluate first line if multi-line output generated
            first_line = cleaned_pred.split("\n")[0].strip()

            is_corr = dataset_mod.evaluate_answer(cleaned_pred, sample.answer, sample.meta)
            if not is_corr and first_line != cleaned_pred:
                is_corr = dataset_mod.evaluate_answer(first_line, sample.answer, sample.meta)

            if is_corr:
                correct += 1

            # Track hops if present in metadata
            hops = sample.meta.get("hops", None)
            if hops is not None:
                if hops not in hop_stats:
                    hop_stats[hops] = {"total": 0, "correct": 0}
                hop_stats[hops]["total"] += 1
                if is_corr:
                    hop_stats[hops]["correct"] += 1

            records.append({
                "sample_id": sample.meta.get("sample_id", len(records)),
                "question": sample.question,
                "target_answer": sample.answer,
                "generated_prediction": cleaned_pred,
                "is_correct": is_corr,
                "meta": sample.meta
            })

    elapsed = time.time() - start_time
    accuracy = (correct / total) * 100.0 if total > 0 else 0.0

    print("\n" + "=" * 60)
    print(f"Direct Evaluation Result (No CoT Baseline)")
    print("=" * 60)
    print(f"Model:           {model_name}")
    print(f"Dataset:         {data_cfg.get('type')} ({split})")
    print(f"Total Samples:   {total}")
    print(f"Correct Samples: {correct}")
    print(f"Direct Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"Time Taken:      {elapsed:.2f}s ({elapsed / max(1, total):.4f}s / sample)")

    if hop_stats:
        print("\nAccuracy by Reasoning Hops:")
        for h in sorted(hop_stats.keys()):
            h_tot = hop_stats[h]["total"]
            h_corr = hop_stats[h]["correct"]
            h_acc = (h_corr / h_tot) * 100.0 if h_tot > 0 else 0.0
            print(f"  Hops = {h}: {h_acc:.2f}% ({h_corr}/{h_tot})")
    print("=" * 60)

    # 4. Save Results
    out_dir = Path("runs/baselines")
    out_dir.mkdir(parents=True, exist_ok=True)
    d_name = data_cfg.get("type", "data")
    out_file = cfg.get("output_file", None)
    if out_file is None:
        safe_model = model_name.replace("/", "_")
        out_file = out_dir / f"direct_{safe_model}_{d_name}_{split}.json"
    else:
        out_file = Path(out_file)

    report = {
        "model": model_name,
        "dataset": d_name,
        "split": split,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "accuracy_ratio": correct / max(1, total),
        "time_seconds": elapsed,
        "hop_stats": hop_stats,
        "predictions": records
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n[DirectEval] Full report and predictions saved to: {out_file}\n")
    return report

if __name__ == "__main__":
    evaluate_backbone_direct()
