#!/usr/bin/env python3
import sys
import copy
from pathlib import Path
import subprocess

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from lsrr.config import load_config_with_cli, expand_sweep_configs, save_config_snapshot

def run_sweep():
    base_cfg = load_config_with_cli()

    # Determine base experiment name from CLI or config
    exp_name = "sweep_exp"
    for arg in sys.argv:
        if arg.startswith("exp="):
            exp_name = arg.split("=")[1]

    child_configs = expand_sweep_configs(base_cfg, base_name=exp_name)
    print(f"[Sweep] Generated {len(child_configs)} child experiments to run.")

    for idx, (child_name, child_conf) in enumerate(child_configs):
        print(f"\n==========================================")
        print(f"[Sweep {idx+1}/{len(child_configs)}] Starting: {child_name}")
        print(f"==========================================")

        # Save temporary resolved config for child
        tmp_cfg_path = Path("runs") / "_sweep_configs" / f"{child_name}.yaml"
        save_config_snapshot(child_conf, tmp_cfg_path)

        cmd = [
            sys.executable,
            str(repo_root / "scripts" / "train.py"),
            f"config={tmp_cfg_path}",
            f"exp_name={child_name}"
        ]
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print(f"[Sweep] Error: Child experiment '{child_name}' failed with returncode {ret.returncode}")

    print("\n[Sweep] All sweep experiments completed!")

if __name__ == "__main__":
    run_sweep()
