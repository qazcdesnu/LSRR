"""스윕 전개.

`sweep:` 절의 데카르트 곱을 자식 설정들로 펼친다. 실험은 설정 조합이며,
새 실험을 위해 새 파이썬 파일을 만들어야 한다면 슬롯 설계가 틀린 것이다 (ADR-009).

이식: Legacy_LSRR/lsrr/config.py:expand_sweep_configs
"""

from __future__ import annotations

import copy
import itertools
from typing import Any, Iterator

from omegaconf import DictConfig, OmegaConf

from lsrr.core.errors import ConfigError


def _short(value: Any) -> str:
    """런 이름에 넣을 수 있게 값을 축약한다."""
    return str(value).replace("/", "_").replace(".", "_").replace(" ", "")


def expand_sweep(cfg: DictConfig, base_name: str = "exp") -> list[tuple[str, DictConfig]]:
    """`sweep:` 절을 (자식 이름, 자식 설정) 목록으로 전개한다.

    `sweep:`가 없으면 `[(base_name, cfg)]` 하나를 반환한다 — 호출부가
    스윕 여부를 분기하지 않아도 되게 하기 위해서다.
    """
    if "sweep" not in cfg or not cfg.sweep:
        return [(base_name, cfg)]

    sweep = OmegaConf.to_container(cfg.sweep, resolve=True)
    if not isinstance(sweep, dict):
        raise ConfigError(f"sweep 절은 매핑이어야 하는데 {type(sweep)}이다.")

    keys = list(sweep.keys())
    value_lists = [v if isinstance(v, list) else [v] for v in (sweep[k] for k in keys)]

    empty = [k for k, v in zip(keys, value_lists) if not v]
    if empty:
        raise ConfigError(f"sweep 축 {empty}의 값 목록이 비어 있다.")

    base = copy.deepcopy(cfg)
    del base["sweep"]

    results: list[tuple[str, DictConfig]] = []
    for combo in itertools.product(*value_lists):
        dotlist = [f"{k}={v}" for k, v in zip(keys, combo)]
        name_parts = [base_name] + [
            f"{k.split('.')[-1]}_{_short(v)}" for k, v in zip(keys, combo)
        ]
        child = OmegaConf.merge(base, OmegaConf.from_dotlist(dotlist))
        results.append(("_".join(name_parts), child))  # type: ignore[arg-type]

    return results


def sweep_size(cfg: DictConfig) -> int:
    """전개될 자식 런 수. 실행 전에 규모를 확인할 때 쓴다."""
    if "sweep" not in cfg or not cfg.sweep:
        return 1
    sweep = OmegaConf.to_container(cfg.sweep, resolve=True)
    n = 1
    for v in sweep.values():  # type: ignore[union-attr]
        n *= len(v) if isinstance(v, list) else 1
    return n


def iter_sweep(cfg: DictConfig, base_name: str = "exp") -> Iterator[tuple[str, DictConfig]]:
    yield from expand_sweep(cfg, base_name)


__all__ = ("expand_sweep", "sweep_size", "iter_sweep")
