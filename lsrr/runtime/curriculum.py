"""단계적 잠재화 커리큘럼 (제안서 v2.1 §5.1, Coconut 계승).

    S1     [X, h⁽¹⁾, c₁ … c_S, y]              M=1 고정,  CoT 제거 0
    S2.k   [X, h⁽¹⁾ … h⁽ᵏ⁺¹⁾, c_{k+1} … c_S, y]  M=k+1 고정, CoT 앞 k 단계 제거
    S3     [X, h⁽¹..M⁾, y]                      M 동적(Δ<ε), CoT 전부 제거

**왜 필요한가 — Coconut Table 1:** 커리큘럼 없는 Coconut 은 76.1% 로 No-CoT(76.7%)
를 넘지 못한다. 백본 전체를 미세조정하고도. 우리 게이트는 그 구성(콜드 스타트)
에서 시험됐고(F-033·F-034), 77 → 97 구간을 여는 것이 이 커리큘럼이다.

**페이즈와 직교한다.** 페이즈(`runtime/phases.py`)는 *무엇을 학습하는가*, 스테이지는
*무엇을 감독하고 M 을 몇으로 두는가* 다. 스테이지는 Phase A 안에서 돈다.

스테이지가 바꾸는 것은 셋뿐이다:
  1. 타깃 — `PromptSpec.cot_keep_from` (남은 CoT + 답). 손실은 그 위치에만.
  2. M — 스케줄·종료를 `fixed`/`fixed_m` 으로 바꾼다. S3 는 설정의 동적 규칙으로 복원.
  3. 옵티마이저 — 스테이지 전환 시 리셋 (Coconut·iCoT 의 "reset optimizer state").
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Optional

from lsrr.core.errors import ConfigError
from lsrr.core.registry import SCHEDULE_REGISTRY, TERMINATION_REGISTRY
from lsrr.data.prompting import PromptSpec


@dataclass(frozen=True)
class StageSpec:
    """커리큘럼 스테이지 하나.

    Attributes:
        name: 기록 이름 (`S1`, `S2.1`, …, `S3`).
        remove_cot: CoT 앞에서 제거할 단계 수. None 이면 **전부** 제거 (S3).
        M: 고정 사이클 수. None 이면 설정의 동적 규칙을 쓴다 (S3).
        epochs: 이 스테이지의 에폭 수.
        max_answer_tokens: 감독 문자열 절단 길이. CoT 가 붙는 스테이지는 길어야 한다.
    """

    name: str
    remove_cot: Optional[int]
    M: Optional[int]
    epochs: int
    max_answer_tokens: int

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ConfigError(f"스테이지 '{self.name}' 의 epochs 는 1 이상이어야 한다.")
        if self.M is not None and self.M < 1:
            raise ConfigError(f"스테이지 '{self.name}' 의 M 은 1 이상이어야 한다.")
        if self.remove_cot is not None and self.remove_cot < 0:
            raise ConfigError(f"스테이지 '{self.name}' 의 remove_cot 는 0 이상이어야 한다.")

    @property
    def is_latent_only(self) -> bool:
        return self.remove_cot is None

    def prompt_spec(self, base: PromptSpec) -> PromptSpec:
        """이 스테이지의 감독 규약. S3 는 base 그대로(답만)."""
        return replace(
            base,
            cot_keep_from=self.remove_cot,
            max_answer_tokens=self.max_answer_tokens,
        )


def stages_from_cfg(cfg: Any) -> list[StageSpec]:
    """`train.curriculum` 절 → 스테이지 목록.

    절이 없으면 빈 목록 — 콜드 스타트(Stage 3 직행)이며 기존 설정이 그대로 돈다.

        train:
          curriculum:
            max_cot_steps: 6         # S2 를 k=1..6 으로 전개
            answer_tokens_with_cot: 96
            stages:
              - {name: S1, remove_cot: 0, M: 1, epochs: 2}
              - {expand: S2, epochs: 2}            # k=1..max_cot_steps, M=k+1
              - {name: S3, epochs: 3}              # remove_cot·M 없음 = 완전 잠재·동적
    """
    node = _get(cfg, "train.curriculum", None)
    if not node:
        return []
    max_k = int(node.get("max_cot_steps", 0))
    cot_tokens = int(node.get("answer_tokens_with_cot", 96))
    answer_tokens = int(_get(cfg, "prompt.max_answer_tokens", 32))
    out: list[StageSpec] = []
    for raw in node.get("stages", []):
        raw = dict(raw)
        if raw.get("expand") == "S2":
            if max_k < 1:
                raise ConfigError("expand: S2 를 쓰려면 curriculum.max_cot_steps ≥ 1 이어야 한다.")
            for k in range(1, max_k + 1):
                out.append(StageSpec(f"S2.{k}", remove_cot=k, M=k + 1,
                                     epochs=int(raw.get("epochs", 1)),
                                     max_answer_tokens=cot_tokens))
            continue
        remove = raw.get("remove_cot")
        latent_only = remove is None
        out.append(StageSpec(
            name=str(raw.get("name", f"stage{len(out)}")),
            remove_cot=None if latent_only else int(remove),
            M=None if raw.get("M") is None else int(raw["M"]),
            epochs=int(raw.get("epochs", 1)),
            max_answer_tokens=answer_tokens if latent_only else cot_tokens,
        ))
    if not out:
        raise ConfigError("train.curriculum.stages 가 비어 있다.")
    return out


def apply_stage(runner: Any, cfg: Any, stage: StageSpec) -> dict[str, Any]:
    """러너의 스케줄·종료를 스테이지에 맞춘다. S3 는 설정값으로 복원한다.

    `M` 을 고정할 때는 `fixed_m` 종료를 같이 두어야 평가 경로도 같은 M 을 쓴다 —
    학습만 고정하고 평가는 동적으로 두면 F-031 과 같은 경로 불일치가 된다.
    """
    if stage.M is None:
        runner.schedule = SCHEDULE_REGISTRY.build(dict(_get(cfg, "recurrence.schedule")))
        runner.termination = TERMINATION_REGISTRY.build(dict(_get(cfg, "termination")))
        return {"M": "dynamic"}
    m_max = int(_get(cfg, "termination.m_max", 32))
    if stage.M > m_max:
        raise ConfigError(f"스테이지 '{stage.name}' 의 M={stage.M} 가 termination.m_max={m_max} 를 넘는다.")
    runner.schedule = SCHEDULE_REGISTRY.build({"type": "fixed", "k": stage.M})
    runner.termination = TERMINATION_REGISTRY.build({"type": "fixed_m", "m": stage.M, "m_max": m_max})
    return {"M": stage.M}


def _get(cfg: Any, path: str, default: Any = None) -> Any:
    from lsrr.config.schema import get_path

    v = get_path(cfg, path, default)
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(v):
            return OmegaConf.to_container(v, resolve=True)
    except Exception:
        pass
    return v


__all__ = ("StageSpec", "stages_from_cfg", "apply_stage")
