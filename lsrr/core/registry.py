"""컴포넌트 레지스트리.

실험은 코드 수정이 아니라 설정의 `type` 교체로 수행된다 (ADR-009).
그 교체가 가능하려면 모든 슬롯 구현이 안정적인 문자열 키로 조회 가능해야 한다.

레지스트리 키는 **설정 파일에 쓰는 이름**이며 클래스 이름과 독립적으로 안정적이다.
한 번 논문 표에 실린 키는 바꾸지 않는다 (CONVENTIONS.md §1.1).

이식: Legacy_LSRR/lsrr/registry.py (시그니처 기반 kwargs 필터링·별칭 등록 계승)
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Iterable, TypeVar

from lsrr.core.errors import RegistryError

T = TypeVar("T")


class Registry:
    """이름 → 구현 클래스/팩토리 매핑.

    Args:
        name: 레지스트리 이름 (슬롯 이름과 일치).
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._registry: dict[str, Any] = {}
        self._canonical: dict[Any, str] = {}

    @property
    def name(self) -> str:
        return self._name

    def register(self, key: str | None = None, override: bool = False) -> Callable[[T], T]:
        """데코레이터로 구현을 등록한다.

        여러 번 적용하면 별칭이 된다. 첫 등록 키가 정본(canonical)이다.

        **중복 키는 거부한다.** 조용한 덮어쓰기는 테스트 더미가 실제 구현을
        가리는 사고를 낳는다 — 실제로 M2에서 conftest의 더미 데이터셋이
        `multiplication` 키를 가로채 빈 split을 반환한 일이 있었다. 의도적
        교체는 `override=True`로 명시한다.
        """

        def decorator(target: T) -> T:
            k = key if key is not None else getattr(target, "__name__", None)
            if k is None:
                raise RegistryError(f"'{self._name}' 레지스트리: 등록 키를 결정할 수 없다.")
            existing = self._registry.get(k)
            if existing is not None and existing is not target and not override:
                raise RegistryError(
                    f"'{k}'는 이미 '{self._name}' 레지스트리에 등록되어 있다 "
                    f"({getattr(existing, '__name__', existing)}). 의도적 교체라면 "
                    f"register(key, override=True)를 쓰라."
                )
            self._registry[k] = target
            self._canonical.setdefault(target, k)
            return target

        return decorator

    def get(self, key: str) -> Any:
        if key not in self._registry:
            raise RegistryError(
                f"'{key}'는 '{self._name}' 레지스트리에 없다. "
                f"등록된 키: {sorted(self._registry)}"
            )
        return self._registry[key]

    def canonical_key(self, key: str) -> str:
        """별칭을 정본 키로 정규화한다 (런 기록·표 표기용)."""
        return self._canonical.get(self.get(key), key)

    def build(self, cfg: Any, **kwargs: Any) -> Any:
        """설정 dict(`{"type": ..., ...}`)로부터 인스턴스를 만든다.

        `kwargs`는 조립 루트가 주입하는 값(d_model, num_layers 등)이며 설정보다
        우선한다. 대상 생성자가 `**kwargs`를 받지 않으면 시그니처에 있는 인자만
        전달한다 — 슬롯마다 필요 없는 주입값을 일일이 걸러내지 않기 위해서다.
        """
        cfg_dict = _to_plain_dict(cfg, self._name)

        if "type" not in cfg_dict:
            raise RegistryError(
                f"'{self._name}' 설정에 'type' 키가 없다. 받은 값: {cfg_dict}"
            )

        type_key = cfg_dict.pop("type")
        target = self.get(type_key)
        merged = {**cfg_dict, **kwargs}

        try:
            sig = inspect.signature(target)
        except (TypeError, ValueError):  # 시그니처를 읽을 수 없는 콜러블
            return target(**merged)

        accepts_var_kw = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        if not accepts_var_kw:
            merged = {k: v for k, v in merged.items() if k in sig.parameters}

        try:
            return target(**merged)
        except TypeError as exc:
            raise RegistryError(
                f"'{self._name}.{type_key}' 생성 실패: {exc}. 전달 인자: {sorted(merged)}"
            ) from exc

    def keys(self) -> list[str]:
        return sorted(self._registry)

    def __contains__(self, key: object) -> bool:
        return key in self._registry

    def __len__(self) -> int:
        return len(self._registry)

    def __repr__(self) -> str:
        return f"Registry({self._name!r}, {len(self._registry)} entries)"


def _to_plain_dict(cfg: Any, registry_name: str) -> dict[str, Any]:
    """설정 객체를 평범한 dict로 변환한다 (DictConfig / dict / str 지원)."""
    if isinstance(cfg, str):
        return {"type": cfg}
    if isinstance(cfg, dict):
        return dict(cfg)

    # omegaconf.DictConfig를 선택적 의존으로 다룬다 — core는 config를 import하지 않는다.
    to_container = getattr(cfg, "_metadata", None)
    if to_container is not None or type(cfg).__name__ == "DictConfig":
        from omegaconf import OmegaConf  # 지역 import: core의 하드 의존을 피한다

        return dict(OmegaConf.to_container(cfg, resolve=True))  # type: ignore[arg-type]

    raise RegistryError(
        f"'{registry_name}' 레지스트리: str/dict/DictConfig가 필요한데 {type(cfg)}를 받았다."
    )


# --- 슬롯별 레지스트리 (ARCHITECTURE.md §3 슬롯 맵과 1:1) ---

BACKBONE_REGISTRY = Registry("backbone")
POOLER_REGISTRY = Registry("pooler")
COMPOSER_REGISTRY = Registry("composer")
SCOPE_REGISTRY = Registry("scope")
ADAPTER_REGISTRY = Registry("adapter")
ENGINE_REGISTRY = Registry("engine")
SCHEDULE_REGISTRY = Registry("schedule")
TERMINATION_REGISTRY = Registry("termination")
STABILITY_REGISTRY = Registry("stability")
FUSION_REGISTRY = Registry("fusion")
READOUT_REGISTRY = Registry("readout")
OBJECTIVE_REGISTRY = Registry("objective")
DATA_REGISTRY = Registry("data")
ANALYSIS_REGISTRY = Registry("analysis")
GATE_REGISTRY = Registry("gate")

ALL_REGISTRIES: dict[str, Registry] = {
    r.name: r
    for r in (
        BACKBONE_REGISTRY,
        POOLER_REGISTRY,
        COMPOSER_REGISTRY,
        SCOPE_REGISTRY,
        ADAPTER_REGISTRY,
        ENGINE_REGISTRY,
        SCHEDULE_REGISTRY,
        TERMINATION_REGISTRY,
        STABILITY_REGISTRY,
        FUSION_REGISTRY,
        READOUT_REGISTRY,
        OBJECTIVE_REGISTRY,
        DATA_REGISTRY,
        ANALYSIS_REGISTRY,
        GATE_REGISTRY,
    )
}


def registry_snapshot() -> dict[str, list[str]]:
    """등록 현황 (런 메타 기록·디버깅용)."""
    return {name: reg.keys() for name, reg in ALL_REGISTRIES.items()}


__all__ = (
    "Registry",
    "ALL_REGISTRIES",
    "registry_snapshot",
    *(f"{r.name.upper()}_REGISTRY" for r in ALL_REGISTRIES.values()),
)
