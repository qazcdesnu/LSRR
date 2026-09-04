import os
import sys
import copy
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import yaml
from omegaconf import OmegaConf, DictConfig

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"

def _find_config_file(name: str, base_dir: Path) -> Path:
    """Resolve a config reference like 'base', 'backbone/gpt2', or direct file path to an actual .yaml path."""
    direct_p = Path(name)
    if direct_p.exists():
        return direct_p.resolve()

    if not name.endswith(".yaml") and not name.endswith(".yml"):
        candidates = [base_dir / f"{name}.yaml", base_dir / f"{name}.yml"]
    else:
        candidates = [base_dir / name]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Config '{name}' not found under {base_dir}. Looked for: {candidates}")

def load_yaml(file_path: Path) -> Dict[str, Any]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}

def resolve_hierarchical_config(config_path: Path, base_dir: Path = None) -> DictConfig:
    """Recursively resolve defaults list and merge them in order."""
    if base_dir is None:
        base_dir = DEFAULT_CONFIG_DIR

    raw_dict = load_yaml(config_path)
    defaults = raw_dict.pop("defaults", [])

    merged = OmegaConf.create({})

    # Merge each default entry
    for def_entry in defaults:
        def_file = _find_config_file(def_entry, base_dir)
        def_conf = resolve_hierarchical_config(def_file, base_dir)
        merged = OmegaConf.merge(merged, def_conf)

    # Merge current file content over defaults
    current_conf = OmegaConf.create(raw_dict)
    merged = OmegaConf.merge(merged, current_conf)

    return merged

def parse_cli_overrides(args: List[str]) -> Tuple[Optional[str], List[str]]:
    """Separate experiment specifier ('exp=name' or 'config=path') from dotlist overrides."""
    exp_name = None
    overrides = []

    for arg in args:
        if arg.startswith("exp="):
            exp_name = arg.split("=", 1)[1]
        elif arg.startswith("config="):
            exp_name = arg.split("=", 1)[1]
        elif "=" in arg:
            overrides.append(arg)

    return exp_name, overrides

def load_config_with_cli(
    args: Optional[List[str]] = None,
    default_exp: Optional[str] = None,
    config_dir: Optional[Path] = None
) -> DictConfig:
    """Load config from exp/config argument and apply all CLI dotlist overrides."""
    if args is None:
        args = sys.argv[1:]
    if config_dir is None:
        config_dir = DEFAULT_CONFIG_DIR

    exp_name, overrides = parse_cli_overrides(args)
    target_exp = exp_name or default_exp

    if not target_exp:
        # Fallback to base.yaml if no experiment specified
        target_path = _find_config_file("base", config_dir)
    else:
        # Check exp/ directory or direct path
        if "/" not in target_exp and not target_exp.endswith(".yaml"):
            exp_candidate = config_dir / "exp" / f"{target_exp}.yaml"
            if exp_candidate.exists():
                target_path = exp_candidate
            else:
                target_path = _find_config_file(target_exp, config_dir)
        else:
            target_path = _find_config_file(target_exp, config_dir)

    base_conf = resolve_hierarchical_config(target_path, config_dir)

    # Apply dotlist overrides
    if overrides:
        cli_conf = OmegaConf.from_dotlist(overrides)
        base_conf = OmegaConf.merge(base_conf, cli_conf)

    return base_conf

def save_config_snapshot(cfg: DictConfig, output_path: Path):
    """Save full resolved configuration to single YAML file for reproducibility."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg))

def expand_sweep_configs(cfg: DictConfig, base_name: str = "exp") -> List[Tuple[str, DictConfig]]:
    """If 'sweep:' section exists, generate Cartesian product of all sweep parameters
    and return list of (child_exp_name, child_cfg).
    """
    if "sweep" not in cfg or not cfg.sweep:
        return [(base_name, cfg)]

    import itertools
    sweep_dict = OmegaConf.to_container(cfg.sweep, resolve=True)
    keys = list(sweep_dict.keys())
    value_lists = [sweep_dict[k] if isinstance(sweep_dict[k], list) else [sweep_dict[k]] for k in keys]

    results = []
    base_clean = copy.deepcopy(cfg)
    del base_clean["sweep"]

    for combination in itertools.product(*value_lists):
        dotlist = []
        name_parts = [base_name]
        for k, v in zip(keys, combination):
            dotlist.append(f"{k}={v}")
            # Simplify name
            short_k = k.split(".")[-1]
            short_v = str(v).replace("/", "_").replace(".", "_")
            name_parts.append(f"{short_k}_{short_v}")

        child_name = "_".join(name_parts)
        override_conf = OmegaConf.from_dotlist(dotlist)
        child_conf = OmegaConf.merge(base_clean, override_conf)
        results.append((child_name, child_conf))

    return results
