"""스윕 전개.

`sweep:` 절의 데카르트 곱을 자식 설정들로 펼친다. 실험은 설정 조합이며,
새 실험을 위해 새 파이썬 파일을 만들어야 한다면 슬롯 설계가 틀린 것이다 (ADR-009).

전개 결과는 **인덱스로 주소지정 가능**하다. `sweep_plan(cfg)[k]`가 항상 같은
조합을 가리키므로, slurm 배열 작업이 `--array=0-N` ↔ `--index $SLURM_ARRAY_TASK_ID`
로 한 작업당 한 자식을 돌릴 수 있다. 이 결정성은 키 순서와 `itertools.product`의
순서에 달려 있으니, 순서를 흔드는 변경(예: 정렬 추가)은 진행 중인 배열 작업을
깨뜨린다.

이식: Legacy_LSRR/lsrr/config.py:expand_sweep_configs
"""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field
from typing import Any, Iterator

from omegaconf import DictConfig, OmegaConf

from lsrr.core.errors import ConfigError


#: 묶음 축의 값에서 조합 이름으로 쓰는 키. 설정 경로가 아니다.
NAME_KEY = "name"


def _short(value: Any) -> str:
    """런 이름에 넣을 수 있게 값을 축약한다."""
    return str(value).replace("/", "_").replace(".", "_").replace(" ", "")


def _axis_point(axis: str, value: Any) -> tuple[dict[str, Any], str]:
    """축의 한 값 → (덮어쓸 키들, 이름 조각).

    값이 **매핑이면 묶음 축**이다 — 그 매핑의 키들을 한꺼번에 덮어쓴다.
    조건이 여러 키의 특정 조합으로만 성립할 때 필요하다. 예를 들어 Ablation A의
    세 조건은 `emission × termination × anchor` 의 데카르트 곱 8가지 중 3가지일
    뿐이어서, 축을 따로 두면 존재하지 않는 조건 5개가 생긴다. 이때 축 이름
    (`condition:` 등)은 설정 경로가 아니라 **이름표**이며, 각 값의 `name:` 이
    런 이름에 들어간다.
    """
    if not isinstance(value, dict):
        return {axis: value}, f"{axis.split('.')[-1]}_{_short(value)}"

    overrides = {k: v for k, v in value.items() if k != NAME_KEY}
    if not overrides:
        raise ConfigError(
            f"sweep 축 '{axis}'의 묶음 값에 덮어쓸 키가 없다: {value!r}"
        )
    label = value.get(NAME_KEY)
    if label is None:
        label = "-".join(f"{k.split('.')[-1]}_{_short(v)}" for k, v in overrides.items())
    return overrides, f"{axis.split('.')[-1]}_{_short(label)}"


@dataclass(frozen=True)
class SweepChild:
    """전개된 자식 런 하나.

    Attributes:
        index: 전개 순서 안에서의 위치. slurm 배열 인덱스가 이것이다.
        name: 런 이름 (`<base>_<축>_<값>_...`). 런 디렉터리 이름의 앞자리가 된다.
        overrides: 이 자식을 만든 `점 표기 키 → 값` 조합. 자식 프로세스에
            그대로 넘길 수 있는 형태다.
        cfg: 덮어쓰기가 반영되고 `sweep:` 절이 제거된 설정.
    """

    index: int
    name: str
    overrides: dict[str, Any]
    cfg: DictConfig = field(repr=False)

    @property
    def dotlist(self) -> list[str]:
        """`key=value` 목록 — 자식 프로세스 인자로 쓴다."""
        return [f"{k}={v}" for k, v in self.overrides.items()]


def sweep_plan(cfg: DictConfig, base_name: str = "exp") -> list[SweepChild]:
    """`sweep:` 절을 자식 목록으로 전개한다.

    `sweep:`가 없으면 원본 설정 하나짜리 목록을 반환한다 — 호출부가 스윕
    여부를 분기하지 않아도 되게 하기 위해서다.
    """
    if "sweep" not in cfg or not cfg.sweep:
        return [SweepChild(index=0, name=base_name, overrides={}, cfg=cfg)]

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

    children: list[SweepChild] = []
    for i, combo in enumerate(itertools.product(*value_lists)):
        overrides: dict[str, Any] = {}
        name_parts = [base_name]
        for axis, value in zip(keys, combo):
            axis_overrides, label = _axis_point(axis, value)
            overrides.update(axis_overrides)
            name_parts.append(label)
        dotlist = [f"{k}={v}" for k, v in overrides.items()]
        child = OmegaConf.merge(base, OmegaConf.from_dotlist(dotlist))
        children.append(
            SweepChild(
                index=i,
                name="_".join(name_parts),
                overrides=overrides,
                cfg=child,  # type: ignore[arg-type]
            )
        )
    return children


def expand_sweep(cfg: DictConfig, base_name: str = "exp") -> list[tuple[str, DictConfig]]:
    """`sweep_plan`의 (이름, 설정) 형태. 기존 호출부 호환용이다."""
    return [(c.name, c.cfg) for c in sweep_plan(cfg, base_name)]


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


__all__ = (
    "SweepChild",
    "NAME_KEY",
    "sweep_plan",
    "expand_sweep",
    "sweep_size",
    "iter_sweep",
)
