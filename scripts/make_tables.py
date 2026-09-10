#!/usr/bin/env python3
import os
import sys
import json
from pathlib import Path
import yaml
import pandas as pd

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

def make_paper_tables(
    index_csv: Path = Path("runs/runs_index.csv"),
    published_yaml: Path = Path("configs/published_numbers.yaml"),
    output_md: Path = Path("runs/paper_comparison_table.md")
):
    print(f"[TableGen] Loading index from: {index_csv}")

    df_runs = pd.DataFrame()
    if index_csv.exists():
        df_runs = pd.read_csv(index_csv)

    published_data = {}
    if published_yaml.exists():
        with open(published_yaml, "r", encoding="utf-8") as f:
            published_data = yaml.safe_load(f)

    # Collect held-out accuracy from every run that has been evaluated. Without this the
    # comparison table lists only published baselines and no number of our own.
    eval_results = []
    runs_dir = index_csv.parent
    if runs_dir.exists():
        for res_path in sorted(runs_dir.glob("*/eval_results.json")):
            try:
                with open(res_path, "r", encoding="utf-8") as f:
                    eval_results.append(json.load(f))
            except Exception as e:
                print(f"[TableGen] Skipping unreadable {res_path}: {e}")

    lines = []
    lines.append("# LSRR Experimental Results & Comparison Table\n")
    lines.append(f"> Generated automatically from `{index_csv}` and `{published_yaml}`.\n")

    # Group runs by experiment / engine
    if not df_runs.empty:
        lines.append("## 1. Local Experiment Runs Summary\n")
        cols_to_show = [c for c in ["exp_name", "engine", "backbone", "dataset", "seed", "best_val_loss", "total_train_time_sec"] if c in df_runs.columns]
        summary_df = df_runs[cols_to_show].drop_duplicates()
        lines.append(summary_df.to_markdown(index=False))
        lines.append("\n")

    # Held-out evaluation results for our own runs
    if eval_results:
        lines.append("## 2. LSRR Held-out Evaluation\n")
        lines.append("| Run | Dataset | Split | N | Accuracy | Avg M | Latency (ms) | FLOPs/sample | Backbone counted |")
        lines.append("| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- |")
        for r in eval_results:
            b = r.get("base", {})
            lines.append(
                f"| {r.get('run_id','')} | {r.get('dataset','')} | {r.get('split','')} | "
                f"{r.get('num_samples','')} | {b.get('accuracy', float('nan')):.3f} | "
                f"{b.get('avg_stopping_cycles', float('nan')):.2f} | "
                f"{b.get('latency_ms_per_sample', float('nan')):.2f} | "
                f"{b.get('flops_total_per_sample', float('nan')):.3e} | "
                f"{'yes' if b.get('flops_includes_backbone') else 'NO'} |"
            )
        lines.append("\n")

        # Per-hop breakdown, the depth-generalisation view the baselines report.
        hop_rows = [(r, b) for r in eval_results if (b := r.get("base", {})).get("accuracy_by_hops")]
        if hop_rows:
            lines.append("### 2b. Accuracy by reasoning hops\n")
            lines.append("| Run | Hops | Accuracy | N |")
            lines.append("| :--- | ---: | ---: | ---: |")
            for r, b in hop_rows:
                for hops, v in b["accuracy_by_hops"].items():
                    lines.append(f"| {r.get('run_id','')} | {hops} | {v['accuracy']:.3f} | {v['n']} |")
            lines.append("\n")

    # Published comparison table
    lines.append("## 3. Benchmark Comparison against Published Baselines\n")
    lines.append("> Baseline FLOPs are reported per token; ours are per sample and per answer")
    lines.append("> token, and include the single frozen-backbone forward pass.\n")
    lines.append("| Model | Accuracy | Source / Notes |")
    lines.append("| :--- | :--- | :--- |")

    # Add external published numbers
    for bench_name, items in published_data.items():
        lines.append(f"| **[{bench_name.upper()}]** | | |")
        for it in items:
            model = it.get("model", "")
            acc = f"{it['accuracy']:.3f}" if it.get("accuracy") is not None else "-"
            source = it.get("source", "")
            lines.append(f"| {model} | {acc} | {source} |")
        # Our own runs on this benchmark, inline for direct comparison
        for r in eval_results:
            if str(r.get("dataset", "")).lower().startswith(bench_name.split("_")[0].lower()):
                b = r.get("base", {})
                lines.append(
                    f"| LSRR-{r.get('engine','')} (Ours, {r.get('split','')}) | "
                    f"{b.get('accuracy', float('nan')):.3f} | this repo, {r.get('run_id','')} |"
                )

    output_md.parent.mkdir(parents=True, exist_ok=True)
    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[TableGen] Table written to {output_md}")
    print("\n" + "\n".join(lines))

if __name__ == "__main__":
    make_paper_tables()
