"""2원화 학습의 페이즈 정의 (제안서 v2.1 §5.0, ADR-014).

    Phase A  동결 백본 · 엔진·어댑터·방출기 학습     → 완전 동결 조건의 체크포인트
    Phase B  엔진 동결 · LoRA + 방출기 학습          → 순수 정렬 이득
    Phase C  (선택) 공동 미세조정                     → 순차 최적화의 공적응 격차

**이 분해가 Ablation B 를 학습 과정 자체에서 산출한다** (§5.0). Phase A 체크포인트가
완전 동결 조건이고 Phase B 증분이 정렬 이득이므로, 별도 실험을 돌릴 필요가 없다.
그러려면 각 페이즈가 **무엇을 학습하는지** 가 한 곳에 명시돼 있어야 한다 — 페이즈별
`requires_grad` 를 호출부가 손으로 켜고 끄면 두 런이 다른 것을 학습해도 알 수 없다.

그룹 이름은 설정이 쓰는 어휘이고, 모델의 속성 이름과 **의도적으로 다르다**:
`memory` 가 `pipeline`·`pooler`·`composer`·`adapter` 를 함께 가리키는 식이다.
설정이 내부 배선 이름에 묶이면 배선을 바꿀 때 모든 실험 설정이 깨진다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

import torch.nn as nn

from lsrr.core.errors import ConfigError

#: 설정 어휘 → 모델 속성. 한 그룹이 여러 모듈을 가리킬 수 있다.
#:
#: 같은 파라미터가 여러 속성에서 보일 수 있다 — `model.pipeline` 이
#: `model.composer`·`model.adapter` 를 자식으로 품고 모델이 셋을 모두 속성으로
#: 노출한다. 그래서 수집은 **반드시 id 로 중복을 제거한다**: 중복이 남으면
#: 옵티마이저가 같은 파라미터를 두 번 갱신해 그 모듈만 유효 학습률이 2배가 된다.
GROUPS: dict[str, tuple[str, ...]] = {
    "engine": ("engine",),
    "memory": ("pipeline", "pooler", "composer", "adapter"),
    "emitter": ("fusion", "readout"),
}
"""`lora` 는 모델 트리 밖(백본 인스턴스 안)에 있어 여기 없다 — 따로 다룬다."""


def _unique(params: Iterable[nn.Parameter]) -> list[nn.Parameter]:
    """등장 순서를 지키며 id 로 중복을 제거한다."""
    seen: set[int] = set()
    out: list[nn.Parameter] = []
    for p in params:
        if id(p) not in seen:
            seen.add(id(p))
            out.append(p)
    return out

LORA_GROUP = "lora"
VALID_GROUPS = frozenset(GROUPS) | {LORA_GROUP}


@dataclass(frozen=True)
class PhaseSpec:
    """한 페이즈의 학습 계획.

    Attributes:
        name: 기록에 남는 이름 (`A`/`B`/`C`).
        epochs: 이 페이즈의 에폭 수.
        lr: 이 페이즈의 최대 학습률. Phase B 는 정렬이므로 대개 A 보다 낮다.
        trainable: 학습할 그룹. 여기 없는 그룹은 **동결된다** — 이전 페이즈에서
            학습 중이었더라도 그렇다.
        attach_lora: 이 페이즈 시작 시 어댑터를 장착할지. 이미 붙어 있으면 무시한다.
    """

    name: str
    epochs: int
    lr: float
    trainable: tuple[str, ...]
    attach_lora: bool = False
    #: 그룹별 lr 재정의. 공동 학습(Ablation E)에서 엔진(3e-4)과 LoRA(1e-4)를 한
    #: lr 로 묶으면 레시피가 교락된다 — 각자 검증된 값을 유지한다.
    lr_groups: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        unknown = [g for g in self.trainable if g not in VALID_GROUPS]
        if unknown:
            raise ConfigError(
                f"페이즈 '{self.name}' 의 trainable 그룹 {unknown} 을 모른다. "
                f"가능: {sorted(VALID_GROUPS)}."
            )
        if self.epochs < 1:
            raise ConfigError(f"페이즈 '{self.name}' 의 epochs 는 1 이상이어야 한다.")
        for g, _ in self.lr_groups:
            if g not in self.trainable:
                raise ConfigError(
                    f"페이즈 '{self.name}' 의 lr_groups 에 학습 대상이 아닌 그룹 '{g}' 이 있다."
                )
        if LORA_GROUP in self.trainable and not self.attach_lora:
            raise ConfigError(
                f"페이즈 '{self.name}' 이 lora 를 학습 대상으로 두면서 "
                f"attach_lora 가 꺼져 있다. 붙이지 않은 어댑터는 학습할 수 없다."
            )


def _modules_of(model: nn.Module, group: str) -> list[nn.Module]:
    return [
        m for attr in GROUPS[group]
        if isinstance(m := getattr(model, attr, None), nn.Module)
    ]


def apply_phase(
    model: nn.Module, encoder: Any, spec: PhaseSpec
) -> dict[str, Any]:
    """페이즈의 `requires_grad` 를 적용하고 최적화 대상을 모은다.

    **모든 그룹을 먼저 끄고 켠다.** 이전 페이즈의 상태를 물려받으면 페이즈 순서에
    따라 학습 대상이 달라진다.

    Returns:
        `{"params": [...], "by_group": {이름: 파라미터 수}}`
    """
    for group in GROUPS:
        for module in _modules_of(model, group):
            for p in module.parameters():
                p.requires_grad_(False)

    params: list[nn.Parameter] = []
    by_group: dict[str, int] = {}
    group_lists: dict[str, list[nn.Parameter]] = {}

    for group in spec.trainable:
        if group == LORA_GROUP:
            continue
        group_params = _unique(
            p for m in _modules_of(model, group) for p in m.parameters()
        )
        for p in group_params:
            p.requires_grad_(True)
        params.extend(group_params)
        by_group[group] = sum(p.numel() for p in group_params)
        group_lists[group] = group_params

    if spec.attach_lora and encoder is not None:
        if not getattr(encoder, "has_lora", False):
            raise ConfigError(
                f"페이즈 '{spec.name}' 이 LoRA 를 요구하는데 어댑터가 장착되지 "
                f"않았다. `attach_phase_lora` 를 먼저 부르라."
            )
        lora_params = encoder.lora_parameters()
        train_lora = LORA_GROUP in spec.trainable
        for p in lora_params:
            p.requires_grad_(train_lora)
        if train_lora:
            params.extend(lora_params)
            by_group[LORA_GROUP] = sum(p.numel() for p in lora_params)
            group_lists[LORA_GROUP] = list(lora_params)

    # 그룹끼리도 겹칠 수 있으므로 마지막에 한 번 더 거른다.
    params = _unique(params)
    if not params:
        raise ConfigError(
            f"페이즈 '{spec.name}' 에 학습할 파라미터가 하나도 없다 "
            f"(trainable={spec.trainable})."
        )
    # 옵티마이저 파라미터 그룹. lr 재정의가 없는 그룹은 lr 을 넣지 않아 Trainer 의
    # 기본 lr 을 따른다. 같은 파라미터가 두 그룹에 들어가지 않도록 id 로 거른다.
    lr_of = dict(spec.lr_groups)
    seen: set[int] = set()
    param_groups: list[dict[str, Any]] = []
    for g, plist in group_lists.items():
        plist = [p for p in plist if id(p) not in seen]
        seen.update(id(p) for p in plist)
        if not plist:
            continue
        entry: dict[str, Any] = {"params": plist, "name": g}
        if g in lr_of:
            entry["lr"] = float(lr_of[g])
        param_groups.append(entry)
    return {
        "params": params,
        "param_groups": param_groups,
        "by_group": by_group,
        "total": sum(p.numel() for p in params),
    }


def attach_phase_lora(encoder: Any, cfg: Any) -> int:
    """설정의 `backbone.lora` 절로 어댑터를 장착한다. 이미 있으면 무연산."""
    if encoder is None or getattr(encoder, "has_lora", False):
        return 0
    node = _get(cfg, "backbone.lora", {}) or {}
    kwargs = {k: node[k] for k in ("r", "alpha", "dropout", "target_modules") if k in node}
    return encoder.attach_lora(**kwargs)


def phases_from_cfg(cfg: Any) -> list[PhaseSpec]:
    """`train.phases` 절 → 페이즈 목록.

    절이 없으면 **단일 페이즈**로 되돌린다 (`train.epochs`·`train.lr` 사용,
    전 그룹 학습). 기존 1-페이즈 설정이 그대로 돌아야 하기 때문이다.
    """
    raw = _get(cfg, "train.phases", None)
    if not raw:
        return [
            PhaseSpec(
                name="A",
                epochs=int(_get(cfg, "train.epochs", 1)),
                lr=float(_get(cfg, "train.lr", 3e-4)),
                trainable=tuple(GROUPS),
            )
        ]

    specs: list[PhaseSpec] = []
    for i, node in enumerate(raw):
        node = dict(node)
        trainable = node.get("trainable")
        if trainable is None:
            raise ConfigError(f"train.phases[{i}] 에 trainable 이 없다.")
        lr_groups = tuple((str(k), float(v)) for k, v in dict(node.get("lr_groups") or {}).items())
        specs.append(
            PhaseSpec(
                name=str(node.get("name", chr(ord("A") + i))),
                epochs=int(node.get("epochs", _get(cfg, "train.epochs", 1))),
                lr=float(node.get("lr", _get(cfg, "train.lr", 3e-4))),
                trainable=tuple(trainable),
                attach_lora=bool(node.get("attach_lora", LORA_GROUP in trainable)),
                lr_groups=lr_groups,
            )
        )
    return specs


def _get(cfg: Any, path: str, default: Any = None) -> Any:
    from lsrr.config.schema import get_path

    return get_path(cfg, path, default)


__all__ = (
    "GROUPS",
    "LORA_GROUP",
    "PhaseSpec",
    "apply_phase",
    "attach_phase_lora",
    "phases_from_cfg",
)
