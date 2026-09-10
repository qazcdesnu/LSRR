"""계층 설정 로드.

병합 순서 (CONVENTIONS.md §2):
    configs/base.yaml → 도메인 조각 → exp/ 또는 ablation/ → CLI dotlist(최우선)

개작: Legacy_LSRR/lsrr/config.py (defaults 재귀 해석·dotlist 병합 계승)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import yaml
from omegaconf import DictConfig, OmegaConf

from lsrr.config.schema import FILE_SECTIONS
from lsrr.core.errors import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "configs"


def _resolve_file(name: str, base_dir: Path) -> Path:
    """'base', 'backbone/gpt2', 직접 경로를 실제 .yaml 경로로 해석한다."""
    direct = Path(name)
    if direct.is_file():
        return direct.resolve()

    if name.endswith((".yaml", ".yml")):
        candidates = [base_dir / name]
    else:
        candidates = [base_dir / f"{name}.yaml", base_dir / f"{name}.yml"]

    for c in candidates:
        if c.is_file():
            return c.resolve()

    raise ConfigError(
        f"설정 '{name}'을 {base_dir} 아래에서 찾지 못했다. 확인한 경로: "
        f"{[str(c) for c in candidates]}"
    )


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}의 최상위는 매핑이어야 하는데 {type(data)}이다.")
    return data


def resolve_hierarchical(
    config_path: Path,
    base_dir: Optional[Path] = None,
    _seen: Optional[set[Path]] = None,
) -> DictConfig:
    """`defaults:` 목록을 재귀 해석해 순서대로 병합한다."""
    base_dir = base_dir or DEFAULT_CONFIG_DIR
    seen = _seen if _seen is not None else set()

    config_path = config_path.resolve()
    if not config_path.is_file():
        raise ConfigError(f"설정 파일을 찾지 못했다: {config_path}")
    if config_path in seen:
        raise ConfigError(f"설정 defaults에 순환이 있다: {config_path}")
    seen = seen | {config_path}

    raw = load_yaml(config_path)
    defaults = raw.pop("defaults", []) or []
    if isinstance(defaults, str):
        defaults = [defaults]

    merged = OmegaConf.create({})
    for entry in defaults:
        child = _resolve_file(str(entry), base_dir)
        merged = OmegaConf.merge(merged, resolve_hierarchical(child, base_dir, seen))

    return OmegaConf.merge(merged, OmegaConf.create(raw))


def parse_cli(args: Sequence[str]) -> tuple[Optional[str], list[str]]:
    """`exp=`/`config=` 지정자와 dotlist 오버라이드를 분리한다."""
    target: Optional[str] = None
    overrides: list[str] = []
    for arg in args:
        if arg.startswith(("exp=", "config=")):
            target = arg.split("=", 1)[1]
        elif "=" in arg:
            overrides.append(arg)
        else:
            raise ConfigError(
                f"인자 '{arg}'를 해석할 수 없다. 'exp=<name>' 또는 'key=value' 형식이어야 한다."
            )
    return target, overrides


def _expand_file_sections(cfg: DictConfig, config_dir: Path) -> DictConfig:
    """`engine: hydra_qs`처럼 문자열로 쓴 섹션을 해당 파일로 확장한다.

    스키마에 선언된 섹션(FILE_SECTIONS)만 확장한다 — 레거시의 암묵 규칙과 달리
    선언되지 않은 섹션에 문자열을 쓰면 조용히 넘어가지 않고 오류가 된다.
    """
    for section in FILE_SECTIONS:
        node = cfg
        parts = section.split(".")
        for p in parts[:-1]:
            node = node.get(p) if node is not None and p in node else None
            if node is None:
                break
        if node is None:
            continue
        leaf = parts[-1]
        if leaf not in node or not isinstance(node[leaf], str):
            continue

        name = node[leaf]
        path = _resolve_file(f"{section.split('.')[0]}/{name}", config_dir)
        loaded = resolve_hierarchical(path, config_dir)
        node[leaf] = loaded[leaf] if leaf in loaded else loaded
    return cfg


def load_config(
    args: Optional[Sequence[str]] = None,
    default_exp: Optional[str] = None,
    config_dir: Optional[Path] = None,
    validate: bool = True,
) -> DictConfig:
    """CLI 인자로부터 완전히 해석된 설정을 만든다.

    Args:
        args: CLI 인자 (기본: sys.argv[1:]).
        default_exp: `exp=`가 없을 때 쓸 실험 이름.
        config_dir: 설정 루트 (기본: <project>/configs).
        validate: 로드 시점 검증 수행 여부. 끄는 것은 테스트 목적에 한한다.
    """
    import sys

    args = list(sys.argv[1:] if args is None else args)
    config_dir = Path(config_dir) if config_dir else DEFAULT_CONFIG_DIR

    target, overrides = parse_cli(args)
    target = target or default_exp

    if target is None:
        path = _resolve_file("base", config_dir)
    elif "/" in target or target.endswith((".yaml", ".yml")):
        path = _resolve_file(target, config_dir)
    else:
        exp_candidate = config_dir / "exp" / f"{target}.yaml"
        abl_candidate = config_dir / "ablation" / f"{target}.yaml"
        if exp_candidate.is_file():
            path = exp_candidate
        elif abl_candidate.is_file():
            path = abl_candidate
        else:
            path = _resolve_file(target, config_dir)

    cfg = resolve_hierarchical(path, config_dir)

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    cfg = _expand_file_sections(cfg, config_dir)  # type: ignore[assignment]

    if validate:
        from lsrr.config.validate import validate_config

        validate_config(cfg)

    return cfg  # type: ignore[return-value]


__all__ = (
    "PROJECT_ROOT",
    "DEFAULT_CONFIG_DIR",
    "load_yaml",
    "resolve_hierarchical",
    "parse_cli",
    "load_config",
)
