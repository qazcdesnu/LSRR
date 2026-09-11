"""Phase별 게이트 집합과 기본 임계값 (ADR-008).

**임계값은 실험 전에 고정된다.** 이 파일의 기본값이 정본이며, 설정으로 덮어쓸
수는 있으나 그 설정은 런 스냅샷에 포함되어 사후 조정이 기록에 남는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Phase0Thresholds:
    """Phase 0 게이트의 임계값 묶음."""

    # ① 붕괴
    min_effective_rank: float = 2.0
    max_mean_similarity: float = 0.95
    min_variance_ratio: float = 0.1
    # ② anytime
    min_improvement: float = 0.02
    min_trend: float = 0.5
    # ③ 킬 스위치
    alpha: float = 0.05
    min_effect_size: float = 0.5
    #: 절대 정확도 차이의 하한. 효과크기만으로는 실질적으로 무의미한 차이를
    #: 막지 못한다 (F-016). 1%p 는 판단이며, 실험 전에 고정한다.
    min_difference: float = 0.01
    #: 조건당 최소 시드 수. 3 에서 5 로 올렸다 — 1자리 곱셈 실측에서 차이
    #: 17.5%p, d=1.31 인데도 p=0.0543 으로 게이트 ③ 이 떨어졌다. 시드 3개는
    #: 조건 간 분산이 다를 때 검정력이 부족하다. 실험 **전에** 고정한다 (ADR-008).
    min_seeds: int = 5
    # ④ 거동
    min_converged_ratio: float = 0.6

    @classmethod
    def from_config(cls, cfg: object) -> "Phase0Thresholds":
        """설정의 `gates.phase0:` 절에서 덮어쓴다. 없는 키는 기본값."""
        node = cfg.get("gates", {}).get("phase0", {}) if hasattr(cfg, "get") else {}
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in dict(node or {}).items() if k in known})


@dataclass(frozen=True)
class PhaseDefinition:
    """한 Phase의 게이트 목록과 킬 스위치 지정."""

    phase: str
    gate_ids: tuple[str, ...]
    kill_switch: str
    description: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)


PHASE_0 = PhaseDefinition(
    phase="phase0",
    gate_ids=("①", "②", "③", "④"),
    kill_switch="③",
    description="GPT-2 + 곱셈 · ProsQA — 반복 정제가 1회 통과보다 나은가",
    notes=(
        "③ 실패 시 재검토 순서: 깊은 감독 과잉(λ_ds·γ 하향) → 감쇠 α → "
        "안정화 사다리 → 메모리 구성(ADR-004) → 엔진 구조.",
        "싼 것부터, 그리고 제안서가 이미 예상한 실패 양식부터다.",
    ),
)

PHASES = {PHASE_0.phase: PHASE_0}


__all__ = ("PHASES", "PHASE_0", "Phase0Thresholds", "PhaseDefinition")
