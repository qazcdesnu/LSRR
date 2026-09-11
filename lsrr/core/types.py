"""컴포넌트 간에 오가는 데이터 타입.

이 타입들이 형제 패키지가 서로를 모른 채 대화하는 유일한 통로다.
새 순환 의존이 필요해 보이면 여기에 필드가 빠졌다는 신호다 (ARCHITECTURE.md §5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import torch

# --- 데이터 ---


@dataclass
class DataSample:
    """통일 데이터 스키마."""

    question: str
    answer: str
    cot_steps: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


# --- 백본 산출물 ---


@dataclass
class ContextBundle:
    """동결 백본 1회 순전파의 전체 산출물 (I2).

    `backbone`과 `memory`가 서로를 모른 채 대화하는 통로다.

    Attributes:
        H_last: [B, L, d_in] 질문 마지막 토큰 위치의 레이어별 은닉 상태.
        H_pool: [B, L, d_in] 질문 전체 어텐션 풀링, 레이어별 (제안서 §4.1, ADR-004).
            풀링을 쓰지 않는 설정에서는 None.
        h_ctx: [B, d_in] 백본 원본 h^(L). 융합 잔차 앵커이며 **어댑터를 통과하지
            않은** 벡터다 (ADR-003). 주입 공간 정합 I8의 근거.
        hidden_stack: [B, L, T, d_in] 레이어×토큰 전체 스택. 학습 대상인 풀러가
            소비한다 (ADR-012). 메모리가 크므로 풀러가 없으면 None으로 둔다.
        kv_cache: 질문 구간 past_key_values. 연속 디코딩용 (§4.4).
        attention_mask: [B, T] 질문 어텐션 마스크. 좌측 패딩 전제 (ADR-011).
        q_len: [B] 질문 실길이. 연속 디코딩의 position_ids 계산에 쓴다.
        meta: backbone_id, 가중치 해시, include_embedding 등.
    """

    H_last: torch.Tensor
    h_ctx: torch.Tensor
    H_pool: Optional[torch.Tensor] = None
    hidden_stack: Optional[torch.Tensor] = None
    kv_cache: Any = None
    attention_mask: Optional[torch.Tensor] = None
    q_len: Optional[torch.Tensor] = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def batch_size(self) -> int:
        return int(self.H_last.shape[0])

    @property
    def num_layers(self) -> int:
        return int(self.H_last.shape[1])

    @property
    def d_in(self) -> int:
        return int(self.H_last.shape[2])


# --- 사이클 축 산출물 ---


@dataclass
class CycleDiagnostics:
    """정제 사이클 1회분의 진단 지표."""

    m: int
    delta_state: float
    kl_div: Optional[float] = None
    entropy: Optional[float] = None
    stopped: Optional[torch.Tensor] = None  # [B] bool
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReadoutResult:
    """판독 경로 1회 호출의 산출물.

    중간 사이클 판독과 최종 판독이 **같은 타입**이라는 점이 중요하다 —
    판독 경로는 전 사이클 공유이며 사이클별 헤드는 금지된다 (I3, 제안서 §5).
    """

    logits: Optional[torch.Tensor] = None  # [B, T_a, V]
    h_fusion: Optional[torch.Tensor] = None  # [B, d_in] — 궤적의 마지막 토큰
    alpha: Optional[torch.Tensor] = None  # [B, L]
    m: Optional[int] = None
    #: [B, M, d_in] 이 판독에 주입된 궤적 전체 (v2.1 §4.4, ADR-015).
    h_thought: Optional[torch.Tensor] = None


@dataclass
class ReasoningTrace:
    """모델 순전파 1회의 전체 기록.

    금지: 여기에 모듈 객체(융합 헤드, 판독 경로 등)를 싣지 않는다.
    레거시가 `model_outputs["fusion_head"]`로 하던 일이며, 손실 함수와
    판독 경로를 암묵 결합시킨다 (ADR-005). 손실이 사이클별 로짓을 원하면
    훅이 **미리 계산해 값으로** 넣는다.
    """

    logits: Optional[torch.Tensor] = None
    R0: Optional[torch.Tensor] = None
    R_star: Optional[torch.Tensor] = None
    h_ctx: Optional[torch.Tensor] = None
    h_fusion: Optional[torch.Tensor] = None
    #: [B, M, d_in] 동적 궤적 방출 (v2.1 §4.4, ADR-015).
    #: v1 단일 벡터는 M=1 인 특수 경우이며 `h_fusion` 이 그 별칭이다.
    h_thought: Optional[torch.Tensor] = None
    alpha: Optional[torch.Tensor] = None
    stopping_cycles: Optional[torch.Tensor] = None  # [B] long
    per_cycle: list[CycleDiagnostics] = field(default_factory=list)
    per_cycle_readout: list[ReadoutResult] = field(default_factory=list)
    supervised_cycles: list[int] = field(default_factory=list)
    tbptt_window: tuple[int, int] = (0, 0)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def num_cycles(self) -> int:
        return len(self.per_cycle)


# --- 비용 ---


@dataclass
class CostReport:
    """비용 회계 (I7).

    보고되는 FLOPs·지연은 백본 1회 순전파를 **포함**한다. 측정 불가 시
    0으로 두지 말고 `flops_includes_backbone=False` 플래그를 세운다.
    """

    flops_backbone: float = 0.0
    flops_memory: float = 0.0
    flops_engine: float = 0.0
    flops_continuation: float = 0.0
    latency_ms: Optional[float] = None
    avg_cycles: Optional[float] = None
    flops_includes_backbone: bool = True

    @property
    def flops_total(self) -> float:
        return (
            self.flops_backbone
            + self.flops_memory
            + self.flops_engine
            + self.flops_continuation
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "flops_backbone": self.flops_backbone,
            "flops_memory": self.flops_memory,
            "flops_engine": self.flops_engine,
            "flops_continuation": self.flops_continuation,
            "flops_total": self.flops_total,
            "latency_ms": self.latency_ms,
            "avg_cycles": self.avg_cycles,
            "flops_includes_backbone": self.flops_includes_backbone,
        }


# --- 게이트 ---


@dataclass
class GateVerdict:
    """단계 게이트 판정 결과 (제안서 §6.3, ADR-008)."""

    name: str
    passed: bool
    value: Optional[float] = None
    threshold: Optional[float] = None
    kill_switch: bool = False
    evidence: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        kill = " [KILL SWITCH]" if (self.kill_switch and not self.passed) else ""
        detail = ""
        if self.value is not None and self.threshold is not None:
            detail = f" (value={self.value:.4g}, threshold={self.threshold:.4g})"
        return f"{mark}{kill} {self.name}{detail}"


@dataclass
class TerminationSignals:
    """종료 규칙이 판정에 쓰는 신호 묶음.

    규칙은 원시 텐서가 아니라 이 타입을 받는다 — 신호 계산을 signals.py로
    공통화해 규칙마다 같은 계산을 다시 짜지 않게 한다.
    """

    delta_state: torch.Tensor  # [B]
    kl_div: Optional[torch.Tensor] = None  # [B]
    entropy: Optional[torch.Tensor] = None  # [B]
    extra: dict[str, Any] = field(default_factory=dict)


__all__: Sequence[str] = (
    "DataSample",
    "ContextBundle",
    "CycleDiagnostics",
    "ReadoutResult",
    "ReasoningTrace",
    "CostReport",
    "GateVerdict",
    "TerminationSignals",
)
