"""로드 시점 설정 검증.

**학습 3시간 뒤에 터지는 설정 오류는 없어야 한다** (CONVENTIONS.md §2).
검증 규칙은 lsrr/config/README.md의 최소 목록에 대응한다.
"""

from __future__ import annotations

from typing import Any, Optional

from lsrr.config.schema import SLOTS, TOP_LEVEL_KEYS, get_path
from lsrr.core.errors import ConfigError


def _require(cond: bool, message: str) -> None:
    if not cond:
        raise ConfigError(message)


def check_top_level_keys(cfg: Any) -> list[str]:
    """알 수 없는 최상위 키를 잡는다 (오타 방지)."""
    unknown = [k for k in cfg.keys() if k not in TOP_LEVEL_KEYS]
    _require(
        not unknown,
        f"알 수 없는 최상위 설정 키: {unknown}. 허용된 키: {sorted(TOP_LEVEL_KEYS)}. "
        f"새 키를 추가하려면 lsrr/config/schema.py의 TOP_LEVEL_KEYS에 선언하라.",
    )
    return unknown


def check_required_slots(cfg: Any) -> None:
    """필수 슬롯이 존재하고 `type`을 갖는지 확인한다."""
    for spec in SLOTS:
        node = get_path(cfg, spec.section)
        if node is None:
            _require(
                not spec.required,
                f"필수 슬롯 '{spec.section}'이 설정에 없다 "
                f"(레지스트리 '{spec.registry}').",
            )
            continue
        if isinstance(node, str):
            continue  # 문자열 축약 참조는 로더가 확장한다
        _require(
            "type" in node,
            f"슬롯 '{spec.section}'에 'type' 키가 없다. 실험은 설정의 type 교체로 "
            f"수행된다 (ADR-009).",
        )


def effective_max_cycles(cfg: Any) -> Optional[int]:
    """스케줄이 실제로 돌릴 수 있는 최대 사이클 수.

    스케줄 종류마다 깊이를 정하는 키가 다르다. `fixed`에서는 `max`가 무의미한데,
    설정 병합으로 상위 파일의 `max`가 남아 있을 수 있다 — 그 값을 그대로 비교하면
    존재하지 않는 제약 위반을 보고하게 된다.
    """
    stype = get_path(cfg, "recurrence.schedule.type")
    if stype == "fixed":
        k = get_path(cfg, "recurrence.schedule.k")
        return int(k) if k is not None else None
    value = get_path(cfg, "recurrence.schedule.max")
    return int(value) if value is not None else None


def check_cycle_budget(cfg: Any) -> None:
    """사이클 예산의 정합성 (I5 관련)."""
    m_max = get_path(cfg, "termination.m_max")
    sched_max = effective_max_cycles(cfg)
    tbptt_k = get_path(cfg, "recurrence.tbptt_k")

    _require(
        m_max is None or int(m_max) >= 1,
        f"termination.m_max는 1 이상이어야 한다 (={m_max}). 제안서 §4.3은 "
        f"M_max 폴백을 필수 명세로 둔다.",
    )
    if m_max is not None and sched_max is not None:
        _require(
            int(m_max) >= int(sched_max),
            f"termination.m_max({m_max}) < 스케줄 실효 최대 사이클({sched_max}). "
            f"학습이 평가 폴백보다 깊게 돌면 평가에서 학습 분포 밖으로 나간다.",
        )

    # 정지 하한 (v2.1 §4.3). 로드 시점에 잡지 않으면 하한이 폴백에 먹혀
    # "의도한 깊이로 돌지 않는다"가 조용히 성립한다.
    m_min = get_path(cfg, "termination.m_min")
    if m_min is not None:
        _require(
            int(m_min) >= 1,
            f"termination.m_min은 1 이상이어야 한다 (={m_min}).",
        )
        if m_max is not None:
            _require(
                int(m_min) <= int(m_max),
                f"termination.m_min({m_min}) > termination.m_max({m_max}). "
                f"하한을 만족하기 전에 I5 폴백이 걸려 의도한 깊이로 돌지 않는다.",
            )
    if tbptt_k is not None and sched_max is not None:
        _require(
            int(tbptt_k) <= int(sched_max),
            f"recurrence.tbptt_k({tbptt_k}) > 스케줄 실효 최대 사이클({sched_max}). "
            f"BPTT 윈도가 최대 사이클 수보다 클 수 없다.",
        )


def check_injection_space(cfg: Any) -> None:
    """융합 출력 폭이 백본 입력 임베딩 폭과 맞는지 확인한다 (I8, ADR-003)."""
    fusion_type = get_path(cfg, "readout.fusion.fusion_type", "residual")
    d_in = get_path(cfg, "backbone.d_in")
    d_out = get_path(cfg, "readout.fusion.d_out")

    if fusion_type == "residual" and d_in is not None and d_out is not None:
        _require(
            int(d_in) == int(d_out),
            f"residual 융합은 h_ctx(d_in={d_in})에 W_r·h_SSM(d_out={d_out})을 "
            f"더한다 — 두 폭이 같아야 한다. h_fusion은 백본 입력 임베딩 공간의 "
            f"벡터여야 한다 (I8, ADR-003).",
        )


def check_deep_supervision(cfg: Any) -> None:
    """깊은 감독이 요구하는 사이클별 판독이 켜져 있는지 확인한다 (ADR-006)."""
    ds = get_path(cfg, "objective.deep_supervision")
    if ds is None or not ds.get("enabled", False):
        return

    per_cycle = get_path(cfg, "recurrence.hooks.readout_per_cycle", False)
    _require(
        bool(per_cycle),
        "objective.deep_supervision.enabled=true인데 "
        "recurrence.hooks.readout_per_cycle=false다. 답-앵커형 깊은 감독은 "
        "중간 사이클 판독을 요구한다 (제안서 §5).",
    )

    gamma = ds.get("gamma", None)
    if gamma is not None:
        _require(
            0.0 < float(gamma) <= 1.0,
            f"deep_supervision.gamma는 (0, 1] 범위여야 한다 (={gamma}). "
            f"w_m = γ^(M−m)로 후반 사이클을 강조한다.",
        )

    n_cycles = ds.get("num_cycles", 1)
    tbptt_k = get_path(cfg, "recurrence.tbptt_k")
    if tbptt_k is not None:
        _require(
            int(n_cycles) <= int(tbptt_k),
            f"deep_supervision.num_cycles({n_cycles}) > tbptt_k({tbptt_k}). "
            f"윈도 밖 사이클은 detach되어 감독 효과가 없다 (ADR-006).",
        )


def check_termination_signal(cfg: Any) -> None:
    """출력 공간 종료 규칙이 사이클별 판독을 요구하는지 확인한다."""
    rule = get_path(cfg, "termination.type")
    if rule in ("kl_output", "entropy_output"):
        per_cycle = get_path(cfg, "recurrence.hooks.readout_per_cycle", False)
        _require(
            bool(per_cycle),
            f"termination.type={rule}은 출력 공간 신호를 쓰므로 사이클별 판독이 "
            f"필요하다. recurrence.hooks.readout_per_cycle=true로 설정하라 "
            f"(Ablation B).",
        )


def check_memory_scope(cfg: Any) -> list[str]:
    """메모리 범위 절제(Ablation A)의 의도된 축퇴를 경고로 알린다."""
    warnings: list[str] = []
    scope = get_path(cfg, "memory.scope.type")
    if scope == "final_only":
        warnings.append(
            "memory.scope.type=final_only: 레이어 축 길이가 1이 되어 엔진의 "
            "스캔이 축퇴한다. Ablation A의 h^(L) 단독 조건이라면 의도된 것이다."
        )
    return warnings


def check_budget_match(cfg: Any) -> list[str]:
    """Ablation C의 예산 정합 설정을 점검한다 (ADR-009).

    기준 엔진 자신도 스윕에 포함되므로(예: budget_match=hydra_qs인 스윕의
    hydra_qs 자식) self-match는 오류가 아니라 무연산이다. 실제 정합 검사는
    파라미터 수를 세야 하므로 조립 시점(engine/budget.py)의 일이다.
    """
    warnings: list[str] = []
    target = get_path(cfg, "engine.budget_match")
    if target is None:
        return warnings
    if target == get_path(cfg, "engine.type"):
        warnings.append(
            f"engine.budget_match={target}가 engine.type과 같다 — 기준 엔진 "
            f"자신이므로 정합은 무연산이다."
        )
    return warnings


def check_experimental_flags(cfg: Any) -> list[str]:
    """I2 예외(재인코딩 외부 루프)를 눈에 띄게 만든다 (ADR-010)."""
    warnings: list[str] = []
    if get_path(cfg, "experimental.reencoding_loop", False):
        warnings.append(
            "experimental.reencoding_loop=true: 불변식 I2(단일 인코딩)의 명시적 "
            "예외다. 이 런의 수치는 별도 절에서만 보고한다 (ADR-010)."
        )
    return warnings


def check_stability_ladder(cfg: Any) -> list[str]:
    """안정화 사다리 활성 여부를 눈에 띄게 만든다 (ADR-007)."""
    warnings: list[str] = []
    stype = get_path(cfg, "stability.type", "none")
    if stype not in (None, "none"):
        warnings.append(
            f"stability.type={stype}: 안정화 사다리가 켜져 있다. 이 런은 "
            f"안정화 없는 런과 같은 셀에서 비교하지 않는다 (ADR-007)."
        )
    return warnings


def check_fixed_m_consistency(cfg: Any) -> list[str]:
    """`termination.type=fixed_m`의 m과 스케줄 깊이가 어긋나면 알린다.

    학습은 스케줄이, 평가는 종료 규칙이 깊이를 정한다. 둘이 다르면 평가가
    학습과 다른 깊이로 도는데, 그것이 의도(anytime 검증)일 수도 있으므로
    실패가 아니라 경고로 둔다.
    """
    warnings: list[str] = []
    if get_path(cfg, "termination.type") != "fixed_m":
        return warnings
    m = get_path(cfg, "termination.m")
    sched = effective_max_cycles(cfg)
    if m is not None and sched is not None and int(m) != int(sched):
        warnings.append(
            f"termination.m({m})과 스케줄 깊이({sched})가 다르다 — 평가가 학습과 "
            f"다른 사이클 수로 돈다. 의도라면 무시하라(anytime 검증)."
        )
    return warnings


VALIDATORS = (
    check_top_level_keys,
    check_required_slots,
    check_cycle_budget,
    check_injection_space,
    check_deep_supervision,
    check_termination_signal,
    check_budget_match,
)

WARNERS = (
    check_fixed_m_consistency,
    check_memory_scope,
    check_experimental_flags,
    check_stability_ladder,
)


def validate_config(cfg: Any, strict: bool = True) -> list[str]:
    """전체 검증을 수행하고 경고 목록을 반환한다.

    Args:
        cfg: 해석 완료된 설정.
        strict: True면 위반 시 ConfigError. False면 경고 목록에만 담는다
            (테스트·탐색 목적).

    Returns:
        경고 문자열 목록. 경고는 실패가 아니지만 런 메타에 기록된다.
    """
    warnings: list[str] = []

    for check in VALIDATORS:
        try:
            result = check(cfg)
            if isinstance(result, list):
                warnings.extend(result)
        except ConfigError:
            if strict:
                raise
            warnings.append(f"[검증 실패] {check.__name__}")

    for warner in WARNERS:
        warnings.extend(warner(cfg) or [])

    return warnings


__all__ = ("validate_config", "effective_max_cycles", "VALIDATORS", "WARNERS")
