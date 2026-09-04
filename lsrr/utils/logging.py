import os
import json
import sqlite3
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
from omegaconf import OmegaConf, DictConfig
from lsrr.config import save_config_snapshot

def get_git_hash() -> str:
    try:
        res = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return res.decode("utf-8").strip()
    except Exception:
        return "non_git_or_unknown"

class ExperimentTracker:
    """Manages experiment directory, diagnostics logging, checkpoints, and append-only index."""
    def __init__(
        self,
        exp_name: str,
        cfg: DictConfig,
        base_runs_dir: Path = Path("runs"),
        seed: Optional[int] = None
    ):
        self.exp_name = exp_name
        self.cfg = cfg
        self.base_runs_dir = Path(base_runs_dir)
        self.base_runs_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"_s{seed}" if seed is not None else ""
        self.run_id = f"{exp_name}_{timestamp}{suffix}"
        self.run_dir = self.base_runs_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Save config snapshot
        self.config_path = self.run_dir / "config.yaml"
        save_config_snapshot(cfg, self.config_path)

        # Git hash
        self.git_hash = get_git_hash()
        with open(self.run_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump({
                "run_id": self.run_id,
                "exp_name": self.exp_name,
                "timestamp": timestamp,
                "git_hash": self.git_hash,
                "seed": seed
            }, f, indent=2)

        self.diag_log_path = self.run_dir / "diagnostics.jsonl"
        self.metrics_path = self.run_dir / "metrics.json"
        self.index_csv_path = self.base_runs_dir / "runs_index.csv"
        self.index_sqlite_path = self.base_runs_dir / "runs_index.db"

    def log_diagnostics(self, step_data: Dict[str, Any]):
        """Append per-cycle or per-step diagnostic dictionary to diagnostics.jsonl."""
        with open(self.diag_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(step_data) + "\n")

    def save_checkpoint(self, state_dict: Dict[str, Any], filename: str = "checkpoint_best.pt"):
        import torch
        torch.save(state_dict, self.run_dir / filename)

    def log_metrics_and_finish(self, metrics: Dict[str, Any]):
        """Write metrics.json and append 1-line summary to global index (CSV and SQLite)."""
        # Save metrics.json
        with open(self.metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        # Prepare summary row
        summary_row = {
            "run_id": self.run_id,
            "exp_name": self.exp_name,
            "timestamp": datetime.now().isoformat(),
            "git_hash": self.git_hash,
            "seed": self.seed,
            "engine": self.cfg.get("engine", {}).get("type", "unknown"),
            "backbone": self.cfg.get("backbone", {}).get("name", "unknown"),
            "dataset": self.cfg.get("data", {}).get("name", "unknown"),
            **metrics
        }

        # 1. Append to CSV
        df_row = pd.DataFrame([summary_row])
        if not self.index_csv_path.exists():
            df_row.to_csv(self.index_csv_path, index=False)
        else:
            df_row.to_csv(self.index_csv_path, mode="a", header=False, index=False)

        # 2. Append to SQLite
        try:
            with sqlite3.connect(self.index_sqlite_path) as conn:
                df_row.to_sql("experiments", conn, if_exists="append", index=False)
        except Exception:
            pass
