"""설정 트리의 스키마.

레거시는 문자열 섹션을 자동으로 파일로 해석하는 암묵 규칙을 가졌다
(v1.0:lsrr/config.py:105-116). 암묵 규칙은 설정 오타를 조용히 삼키므로
여기서는 어떤 섹션이 어떤 디렉터리에서 해석되는지를 명시적으로 선언한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# --- 슬롯 선언 ---------------------------------------------------------


@dataclass(frozen=True)
class SlotSpec:
    """설정 섹션 하나와 레지스트리 하나의 대응.

    Attributes:
        section: 설정 트리에서의 키 경로 (점 표기).
        registry: 이 섹션이 생성하는 레지스트리 이름.
        config_dir: `configs/` 아래에서 문자열 참조를 해석할 디렉터리.
            None이면 문자열 축약 참조를 허용하지 않는다.
        required: 조립에 반드시 필요한가.
    """

    section: str
    registry: str
    config_dir: Optional[str] = None
    required: bool = True


SLOTS: tuple[SlotSpec, ...] = (
    SlotSpec("backbone", "backbone", "backbone"),
    SlotSpec("backbone.pooler", "pooler", None, required=False),
    SlotSpec("memory.composer", "composer", None, required=False),
    SlotSpec("memory.scope", "scope", None),
    SlotSpec("memory.adapter", "adapter", None),
    SlotSpec("engine", "engine", "engine"),
    SlotSpec("recurrence.schedule", "schedule", None),
    SlotSpec("termination", "termination", "termination"),
    SlotSpec("stability", "stability", "stability", required=False),
    SlotSpec("readout.fusion", "fusion", None),
    SlotSpec("readout.path", "readout", None),
    SlotSpec("data", "data", "data"),
)

SLOT_BY_SECTION: dict[str, SlotSpec] = {s.section: s for s in SLOTS}

FILE_SECTIONS: tuple[str, ...] = tuple(
    s.section for s in SLOTS if s.config_dir is not None
)
"""문자열 축약 참조(`engine: hydra_qs`)를 파일로 해석할 수 있는 섹션."""

TOP_LEVEL_KEYS: frozenset[str] = frozenset(
    {
        "seed",
        "deterministic",
        "device",
        "backbone",
        "data",
        "memory",
        "engine",
        "recurrence",
        "termination",
        "stability",
        "readout",
        "objective",
        "train",
        "eval",
        "extract",
        "gates",
        "sweep",
        "experimental",
        "split",
        "prompt",
    }
)
"""허용된 최상위 키. 오타를 조용히 삼키지 않기 위해 화이트리스트로 둔다."""


# --- 기본값 -----------------------------------------------------------


@dataclass
class Defaults:
    """코드가 보증하는 최소 기본값.

    `configs/base.yaml`이 대부분을 덮어쓰지만, 설정 파일 없이 만들어진
    테스트용 설정도 유효하도록 여기에 최소치를 둔다.
    """

    seed: int = 42
    deterministic: bool = True
    device: str = "auto"
    m_max: int = 32
    m_min: int = 1
    tbptt_k: int = 4
    damping_alpha: float = 0.8
    deep_supervision_gamma: float = 0.85


DEFAULTS = Defaults()


# --- 조회 헬퍼 ---------------------------------------------------------


def get_path(cfg: Any, path: str, default: Any = None) -> Any:
    """점 표기 경로로 중첩 설정을 조회한다. 없으면 `default`."""
    node = cfg
    for part in path.split("."):
        if node is None:
            return default
        try:
            if part not in node:
                return default
        except TypeError:
            return default
        node = node[part]
    return node if node is not None else default


def slot_config(cfg: Any, spec: SlotSpec) -> Optional[Any]:
    """슬롯 설정 노드를 꺼낸다."""
    return get_path(cfg, spec.section)


__all__ = (
    "SlotSpec",
    "SLOTS",
    "SLOT_BY_SECTION",
    "FILE_SECTIONS",
    "TOP_LEVEL_KEYS",
    "Defaults",
    "DEFAULTS",
    "get_path",
    "slot_config",
)
