"""조립 루트.

**유일하게 전 슬롯의 구체 타입을 아는 곳이다.** 다른 어떤 모듈도 슬롯을
직접 만들지 않는다 — 그것이 "어떤 부품도 다른 부품의 구체 타입을 모른다"는
컴포넌트화의 실질이다 (ARCHITECTURE.md §5).

개작: Legacy_LSRR/lsrr/model.py (슬롯 조립·단일 d_model 해석 지점 계승,
`_resolve_d_model`의 d_model=d_in 강제는 폐지 — ADR-003)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from omegaconf import DictConfig

from lsrr.config.schema import SLOT_BY_SECTION, get_path
from lsrr.core.device import resolve_device
from lsrr.core.errors import AssemblyError, ConfigError
from lsrr.core.registry import ALL_REGISTRIES

# 슬롯 구현을 레지스트리에 등록한다. 조립 루트가 유일하게 전 슬롯을 아는 곳이므로
# 여기서 한 번에 import한다 (ARCHITECTURE.md §5).
import lsrr.backbone  # noqa: F401
import lsrr.data  # noqa: F401
import lsrr.engine  # noqa: F401
import lsrr.memory  # noqa: F401
import lsrr.readout  # noqa: F401
import lsrr.recurrence  # noqa: F401
import lsrr.termination  # noqa: F401


@dataclass
class SlotBundle:
    """조립된 슬롯들.

    `model.py`가 이 묶음을 받아 순전파를 엮는다. 슬롯이 늘면 여기에 필드가
    늘고, 그 사실이 ARCHITECTURE.md §3 슬롯 맵과 대조된다.
    """

    encoder: Any = None
    pooler: Any = None
    composer: Any = None
    scope: Any = None
    adapter: Any = None
    engine: Any = None
    schedule: Any = None
    termination: Any = None
    stability: Any = None
    fusion: Any = None
    readout: Any = None
    data: Any = None
    pipeline: Any = None
    runner: Any = None
    objective: Any = None
    device: Any = None
    widths: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            k: v
            for k, v in self.__dict__.items()
            if k not in ("widths", "warnings") and v is not None
        }


def resolve_widths(cfg: DictConfig, d_in: int, num_layers: int) -> dict[str, int]:
    """폭을 **한 곳에서** 해석한다.

    d_model을 슬롯마다 따로 해석하면 조립 시점에 조용히 어긋난다
    (LEGACY_MAP.md §3). d_model은 d_in과 독립이며(ADR-003), 명시가 없으면
    d_in을 따른다 — 엔진 폭을 줄여 §5의 파라미터 예산을 지키는 주된 노브다.
    """
    d_model = get_path(cfg, "memory.adapter.d_model")
    d_model = int(d_model) if d_model is not None else int(d_in)
    return {"d_in": int(d_in), "d_model": d_model, "num_layers": int(num_layers)}


def _build_slot(
    section: str,
    cfg: DictConfig,
    *,
    required: Optional[bool] = None,
    **kwargs: Any,
) -> Any:
    """스키마에 선언된 슬롯 하나를 만든다."""
    spec = SLOT_BY_SECTION.get(section)
    if spec is None:
        raise ConfigError(f"슬롯 '{section}'이 schema.SLOTS에 선언되지 않았다.")

    node = get_path(cfg, section)
    if node is None:
        if required if required is not None else spec.required:
            raise AssemblyError(f"필수 슬롯 '{section}'이 설정에 없다.")
        return None

    stype = node if isinstance(node, str) else node.get("type")
    if stype in (None, "none"):
        return None

    registry = ALL_REGISTRIES[spec.registry]
    return registry.build(node, **kwargs)


def build_slots(
    cfg: DictConfig,
    d_in: Optional[int] = None,
    num_layers: Optional[int] = None,
    build_encoder: bool = True,
) -> SlotBundle:
    """설정으로부터 전 슬롯을 인스턴스화한다.

    Args:
        cfg: 해석·검증 완료된 설정.
        d_in, num_layers: 백본 폭·레이어 수. None이면 인코더를 만들어 물어본다.
        build_encoder: False면 인코더 없이 조립한다 (엔진 전용 실험·테스트).

    Returns:
        SlotBundle. 폭 불일치는 여기서 즉시 실패한다.
    """
    bundle = SlotBundle()

    # 디바이스는 조립 루트가 한 번 정해 모든 슬롯에 내려보낸다. 백본 세션이
    # 스스로 판단하게 두면 학습 모듈과 다른 디바이스에 올라간다.
    device = resolve_device(str(get_path(cfg, "device", "auto")))
    bundle.device = device

    if build_encoder:
        bundle.encoder = _build_slot("backbone", cfg, device=str(device))
        if bundle.encoder is not None:
            d_in = d_in if d_in is not None else bundle.encoder.hidden_dim
            num_layers = num_layers if num_layers is not None else bundle.encoder.num_layers

    if d_in is None or num_layers is None:
        raise AssemblyError(
            "d_in/num_layers를 결정할 수 없다. 인코더를 만들거나 값을 직접 넘겨라."
        )

    widths = resolve_widths(cfg, d_in, num_layers)
    bundle.widths = widths
    d_model = widths["d_model"]

    bundle.pooler = _build_slot("backbone.pooler", cfg, d_in=d_in)
    bundle.composer = _build_slot("memory.composer", cfg, d_in=d_in, num_layers=num_layers)
    bundle.scope = _build_slot("memory.scope", cfg, num_layers=num_layers)

    scoped_layers = num_layers
    if bundle.scope is not None:
        scoped_layers = len(bundle.scope.layer_indices(num_layers))
        widths["scoped_layers"] = scoped_layers

    bundle.adapter = _build_slot(
        "memory.adapter", cfg, d_in=d_in, d_model=d_model, num_layers=scoped_layers
    )
    if bundle.adapter is not None:
        adapter_d = int(getattr(bundle.adapter, "d_model", d_model))
        if adapter_d != d_model:
            raise AssemblyError(
                f"어댑터가 d_model={adapter_d}를 내놓는데 조립은 d_model={d_model}로 "
                f"해석했다. 엔진·융합이 모두 해석된 폭으로 만들어지므로 어댑터가 "
                f"맞춰야 한다."
            )

    bundle.engine = _build_slot("engine", cfg, d_model=d_model, num_layers=scoped_layers)
    bundle.schedule = _build_slot("recurrence.schedule", cfg)
    bundle.termination = _build_slot("termination", cfg)
    bundle.stability = _build_slot("stability", cfg, required=False)

    if bundle.stability is not None and bundle.engine is not None:
        bundle.stability.apply(bundle.engine)
        bundle.warnings.append(
            f"안정화 사다리 활성: {get_path(cfg, 'stability.type')} (ADR-007)"
        )

    # 융합은 d_model 공간의 사고 표현을 d_in 공간으로 되돌린다 (ADR-003, I8).
    bundle.fusion = _build_slot(
        "readout.fusion", cfg, d_model=d_model, d_out=d_in, num_layers=scoped_layers
    )
    answer_head = getattr(bundle.encoder, "answer_head", None)
    bundle.readout = _build_slot(
        "readout.path",
        cfg,
        fusion=bundle.fusion,
        answer_head=answer_head,
        d_model=d_model,
        d_in=d_in,
    )
    bundle.data = _build_slot("data", cfg, required=False)

    # 메모리 파이프라인: compose → scope → adapt 순서를 고정한다
    from lsrr.memory import LayerMemoryPipeline

    bundle.pipeline = LayerMemoryPipeline(
        composer=bundle.composer, scope=bundle.scope, adapter=bundle.adapter
    )

    # 사이클 러너: 엔진 + 스케줄 + 종료 규칙
    if bundle.engine is not None:
        from lsrr.recurrence import CycleRunner

        bundle.runner = CycleRunner(
            engine=bundle.engine,
            schedule=bundle.schedule,
            termination=bundle.termination,
            tbptt_k=int(get_path(cfg, "recurrence.tbptt_k", 4)),
        )

    # 손실 합성
    objective_cfg = get_path(cfg, "objective")
    if objective_cfg is not None:
        from lsrr.objectives import CompositeObjective

        bundle.objective = CompositeObjective.from_config(objective_cfg)

    return bundle


def build_model(
    cfg: DictConfig,
    d_in: Optional[int] = None,
    num_layers: Optional[int] = None,
    build_encoder: bool = True,
) -> Any:
    """전 슬롯을 조립해 `LSRRModel`을 만든다."""
    from lsrr.model import LSRRModel

    bundle = build_slots(cfg, d_in=d_in, num_layers=num_layers, build_encoder=build_encoder)
    return LSRRModel(bundle=bundle, cfg=cfg, runner=bundle.runner)


__all__ = ("SlotBundle", "build_slots", "build_model", "resolve_widths")
