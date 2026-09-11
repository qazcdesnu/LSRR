"""LSRR 전용 예외 타입.

규약(CONVENTIONS.md §3): 조용한 실패 금지. 값이 없으면 추정하지 말고 던진다.
불변식 위반은 경고가 아니라 예외다.
"""

from __future__ import annotations


class LSRRError(Exception):
    """모든 LSRR 예외의 기반."""


# --- 불변식 위반 (ARCHITECTURE.md §4) ---


class InvariantViolation(LSRRError):
    """아키텍처 불변식 위반. 실험 결과를 무효로 만드는 종류의 오류."""

    invariant: str = "?"

    def __init__(self, message: str) -> None:
        super().__init__(f"[{self.invariant}] {message}")


class FrozenBackboneViolation(InvariantViolation):
    """I1: 백본 파라미터가 학습되었거나 가중치가 변했다."""

    invariant = "I1"


class EncodingNotBaseOnly(InvariantViolation):
    """I9: 인코딩 패스가 순수 base 가중치로 수행되지 않았다.

    LoRA 는 디코딩 전용이다 (ADR-014). 인코딩에 어댑터가 새면 `H` 가 조용히
    오염되고 — 학습은 정상적으로 돌므로 — 결과만 설명 불가가 된다.
    """

    invariant = "I9"


class MultipleEncodeError(InvariantViolation):
    """I2: 샘플 배치당 백본 인코딩이 1회를 초과했다.

    사이클 루프 안에서 백본에 재진입하면 Coconut형 반복 호출이 되어
    본 연구의 효율 주장이 사라진다. 재인코딩 외부 루프는 ADR-010의
    명시적 예외 플래그 하에서만 허용된다.
    """

    invariant = "I2"


class SharedReadoutViolation(InvariantViolation):
    """I3: 사이클마다 다른 판독 경로 인스턴스가 쓰였다."""

    invariant = "I3"


class GradientPathViolation(InvariantViolation):
    """I4: 엔진으로의 역전파가 h_fusion 주입 위치가 아닌 통로로 흘렀다."""

    invariant = "I4"


class TerminationFallbackMissing(InvariantViolation):
    """I5: 종료 규칙이 M_max 폴백을 갖지 않는다."""

    invariant = "I5"


class LeakageError(InvariantViolation):
    """I6: 정답이 문맥 인코딩 입력에 노출되었거나 패딩이 감독되었다."""

    invariant = "I6"


class CostAccountingError(InvariantViolation):
    """I7: 비용 보고에 백본 순전파가 누락되었고 플래그도 없다."""

    invariant = "I7"


class InjectionSpaceError(InvariantViolation):
    """I8: h_fusion이 백본 입력 임베딩 공간의 벡터가 아니다."""

    invariant = "I8"


# --- 설정·조립 ---


class ConfigError(LSRRError):
    """설정 로드·검증 실패. 학습 3시간 뒤가 아니라 로드 시점에 던진다."""


class RegistryError(LSRRError):
    """레지스트리 조회·생성 실패."""


class AssemblyError(LSRRError):
    """슬롯 조립 실패 (폭 불일치 등)."""


# --- 데이터·평가 ---


class MissingTargetError(LSRRError):
    """정답이 없는 배치.

    건너뛰면 자유형 데이터셋에서 100%, 수치형에서 0%가 조용히 보고된다.
    치명적 실패로 던진다 (v1.0 교훈, LEGACY_MAP.md §3).
    """


class ScoringError(LSRRError):
    """채점 프로토콜 위반."""
