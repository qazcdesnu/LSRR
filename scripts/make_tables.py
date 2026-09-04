#!/usr/bin/env python3
import os
import sys
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

    # Published comparison table
    lines.append("## 2. Benchmark Comparison against Published Baselines\n")
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

    output_md.parent.mkdir(parents=True, exist_ok=True)
    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[TableGen] Table written to {output_md}")
    print("\n" + "\n".join(lines))

if __name__ == "__main__":
    make_paper_tables()
