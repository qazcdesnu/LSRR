"""전 슬롯의 추상 기반 클래스.

각 인터페이스는 ARCHITECTURE.md §3 슬롯 맵의 한 행에 대응하며, 레지스트리
하나와 짝을 이룬다. 여기에 구체 구현이나 텐서 연산이 들어가서는 안 된다.

텐서 모양 표기 (CONVENTIONS.md §1.5):
    B=배치, L=레이어 축, T=토큰 축, d_in=백본 폭, d_model=사고 메모리 폭,
    M=정제 사이클 수, m=사이클 인덱스(0-indexed)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Protocol, Sequence, runtime_checkable

import torch
import torch.nn as nn

from lsrr.core.types import (
    ContextBundle,
    CycleDiagnostics,
    DataSample,
    GateVerdict,
    ReadoutResult,
    ReasoningTrace,
    TerminationSignals,
)

# ---------------------------------------------------------------- 백본 (L1)


class BaseContextEncoder(ABC):
    """동결 백본. 질문을 **단 1회** 인코딩한다 (I2).

    사이클 루프 안에서 호출되어서는 안 된다. 재인코딩 외부 루프는 ADR-010의
    명시적 예외 플래그 하에서만 허용된다.
    """

    @abstractmethod
    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> ContextBundle:
        """[B, T] → ContextBundle (H_last, H_pool, h_ctx, kv_cache)."""

    @property
    @abstractmethod
    def num_layers(self) -> int:
        """레이어 축 길이 L."""

    @property
    @abstractmethod
    def hidden_dim(self) -> int:
        """백본 폭 d_in."""

    @abstractmethod
    def weight_hash(self) -> str:
        """가중치 해시 (I1: 학습 전후 동일해야 한다)."""


class BaseContextPooler(nn.Module, ABC):
    """레이어별 질문 전체 풀링 (제안서 §4.1, ADR-004).

    단일 토큰의 정보 병목을 막기 위한 글로벌 문맥 보완항을 만든다.
    """

    @abstractmethod
    def forward(
        self,
        hidden_stack: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """[B, L, T, d_in] → [B, L, d_in]."""


class BaseAnswerHead(ABC):
    """백본 연속 디코딩 (제안서 §4.4, ADR-001).

    학습 가능한 디코더가 아니다 — 생성 능력은 끝까지 백본 소유다.
    """

    @abstractmethod
    def teacher_forced(
        self,
        h_fusion: torch.Tensor,
        kv_cache: Any,
        answer_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """[B, d_in] + 질문 KV + [B, T_a] → 로짓 [B, T_a, V]."""

    @abstractmethod
    def generate(
        self,
        h_fusion: torch.Tensor,
        kv_cache: Any,
        max_new_tokens: int = 32,
        eos_token_id: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """탐욕적 자기회귀 생성 → [B, T_gen]."""


# ---------------------------------------------------------------- 메모리 (L1)


class BaseMemoryComposer(nn.Module, ABC):
    """H_last와 H_pool을 결합해 H를 만든다 (ADR-004)."""

    @abstractmethod
    def forward(
        self, H_last: torch.Tensor, H_pool: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """[B, L, d_in] (+ [B, L, d_in]) → [B, L, d_in]."""


class BaseMemoryScope(nn.Module, ABC):
    """레이어 범위 선택 — Ablation A (메모리 범위)."""

    @abstractmethod
    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """[B, L, d_in] → [B, L', d_in]."""

    @abstractmethod
    def layer_indices(self, num_layers: int) -> list[int]:
        """선택된 원본 레이어 인덱스. 분석에서 층별 귀속에 쓴다."""


class BaseLayerAdapter(nn.Module, ABC):
    """레이어별 아핀 변환 + 정규화 (제안서 §4.1).

    층간 표현 공간은 정렬되어 있지 않으므로(tuned lens가 레이어별 변환기를
    필요로 한 이유) 그대로 스캔하면 레이어 축이 의미 있는 시퀀스가 아니다.
    """

    @abstractmethod
    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """[B, L', d_in] → R0 [B, L', d_model]."""

    @property
    @abstractmethod
    def d_model(self) -> int:
        """사고 메모리 폭."""


# ---------------------------------------------------------------- 엔진 (L1)


class BaseRefinementEngine(nn.Module, ABC):
    """레이어 축 정제 연산자 (제안서 §4.2).

    **엔진은 루프를 소유하지 않는다.** `m`은 조건 입력일 뿐이며, 몇 번 도는지·
    언제 멈추는지·언제 detach하는지는 전부 `recurrence`와 `termination`의 일이다.
    이 경계가 §4.2 "축의 구분"의 코드적 표현이다.
    """

    @abstractmethod
    def forward_step(
        self, R_m: torch.Tensor, R0: torch.Tensor, m: int
    ) -> torch.Tensor:
        """정제 1사이클: [B, L, d_model] → [B, L, d_model]."""


# ------------------------------------------------------- 사이클 축 (L1/L2)


class BaseCycleSchedule(ABC):
    """학습 시 사이클 수 M 샘플링과 깊은 감독 가중."""

    @abstractmethod
    def sample_M(self) -> int:
        """이번 스텝의 사이클 수. 로그정규 샘플링이 기본 (제안서 §5)."""

    @abstractmethod
    def weights(self, M: int) -> list[float]:
        """w_m = γ^(M−m) — 후반 사이클 강조 (ADR-006)."""


@runtime_checkable
class CycleHook(Protocol):
    """사이클별 콜백.

    깊은 감독은 손실이 모듈 객체를 꺼내 쓰는 방식이 아니라, 훅이 미리 판독해
    값으로 넣어 주는 방식이다 (ADR-005).
    """

    def on_cycle(
        self,
        m: int,
        R_m: torch.Tensor,
        R_next: torch.Tensor,
        diagnostics: CycleDiagnostics,
    ) -> None: ...


class BaseCycleRunner(ABC):
    """사이클 축 제어. 엔진 내부도, 종료 규칙 내부도 모른다."""

    @abstractmethod
    def run_train(
        self,
        R0: torch.Tensor,
        hooks: Sequence[CycleHook] = (),
        M: Optional[int] = None,
    ) -> ReasoningTrace:
        """M 샘플링 + 절단 BPTT로 학습 순전파."""

    @abstractmethod
    def run_eval(
        self, R0: torch.Tensor, hooks: Sequence[CycleHook] = ()
    ) -> ReasoningTrace:
        """동적 종료로 평가 순전파. m=M_max에서 반드시 정지 (I5)."""


class BaseTerminationRule(ABC):
    """수렴 기반 종료 판정 — Ablation B.

    판정은 **샘플별**이다. 배치 평균으로 정지를 결정하면 "문제 난이도에 따른
    적응적 계산"이라는 주장이 성립하지 않는다.
    """

    m_max: int

    @abstractmethod
    def reset(self, batch_size: int, device: torch.device) -> None: ...

    @abstractmethod
    def should_stop(
        self, signals: TerminationSignals, m: int
    ) -> tuple[torch.Tensor, CycleDiagnostics]:
        """→ (정지 마스크 [B] bool, 진단). M_max 폴백은 기반 클래스가 강제한다."""

    @abstractmethod
    def needs_logits(self) -> bool:
        """출력 공간 신호(KL·엔트로피)를 쓰는가. True면 사이클별 판독이 필요하다."""


class BaseStabilityConstraint(ABC):
    """수렴 안정화 사다리 (ADR-007). 기본은 `none`이며 꺼져 있을 때 흔적이 없다."""

    @abstractmethod
    def apply(self, engine: nn.Module) -> None:
        """파라미터화 개입 (스펙트럴 노름·monotone 재파라미터화)."""

    @abstractmethod
    def penalty(self, R_m: torch.Tensor, R_next: torch.Tensor) -> torch.Tensor:
        """손실 항 (Jacobian 정규화). 비활성 시 0 스칼라."""


# ---------------------------------------------------------------- 판독 (L2)


class BaseFusionHead(nn.Module, ABC):
    """레이어 축 어텐션 풀링 + 잔차 융합 (제안서 §4.4).

        alpha_l = softmax(w^T r_l*),  h_SSM = sum_l alpha_l r_l*
        h_fusion = h_ctx + W_r * h_SSM

    `h_ctx`는 **어댑터를 통과하지 않은** 백본 원본 h^(L)이다 (ADR-003).
    """

    @abstractmethod
    def forward(
        self, R_star: torch.Tensor, h_ctx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """([B, L, d_model], [B, d_in]) → (h_fusion [B, d_in], alpha [B, L])."""


class BaseReadoutPath(nn.Module, ABC):
    """전 사이클 공유 판독 경로 (I3, 제안서 §5).

    `R`은 최종 상태 R*일 수도 중간 사이클 상태 R^(m)일 수도 있으며 **경로는
    동일하다**. 사이클별 헤드 금지 — 임의 시점 종료에도 답이 판독되는 anytime
    성질의 원천이다.
    """

    @abstractmethod
    def readout(
        self,
        R: torch.Tensor,
        h_ctx: torch.Tensor,
        context: ContextBundle,
        answer_ids: Optional[torch.Tensor] = None,
        m: Optional[int] = None,
    ) -> ReadoutResult:
        """정제 상태에서 답을 판독한다."""


# ---------------------------------------------------------------- 손실 (L3)


class BaseObjective(nn.Module, ABC):
    """손실 항. **값만 받는다** — 모델 모듈을 호출하지 않는다 (ADR-005)."""

    @abstractmethod
    def forward(
        self, trace: ReasoningTrace, batch: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        """최소 `"loss"` 키를 포함하는 dict를 반환한다."""


# ------------------------------------------------------ 데이터·분석·게이트


class BaseDataModule(ABC):
    """데이터셋 모듈. 한 샘플의 정오 판정까지가 이 인터페이스의 책임이다."""

    @abstractmethod
    def get_split(self, split: str) -> list[DataSample]: ...

    @abstractmethod
    def score(self, prediction: str, target: str, meta: dict[str, Any]) -> bool:
        """데이터셋별 정규화 후 exact match."""


class BaseAnalysisPlugin(ABC):
    """사후 메커니즘 분석 (제안서 §7). 학습 경로에 개입하지 않는다."""

    @abstractmethod
    def analyze(
        self, traces: Sequence[dict[str, Any]], run_dir: str
    ) -> dict[str, Any]: ...


class BasePhaseGate(ABC):
    """단계 게이트 판정 (제안서 §6.3, ADR-008).

    기준은 실험 **전에** 설정으로 고정하고 런 스냅샷에 포함한다.
    """

    @abstractmethod
    def evaluate(self, run_records: Sequence[dict[str, Any]]) -> GateVerdict: ...


__all__ = (
    "BaseContextEncoder",
    "BaseContextPooler",
    "BaseAnswerHead",
    "BaseMemoryComposer",
    "BaseMemoryScope",
    "BaseLayerAdapter",
    "BaseRefinementEngine",
    "BaseCycleSchedule",
    "BaseCycleRunner",
    "CycleHook",
    "BaseTerminationRule",
    "BaseStabilityConstraint",
    "BaseFusionHead",
    "BaseReadoutPath",
    "BaseObjective",
    "BaseDataModule",
    "BaseAnalysisPlugin",
    "BasePhaseGate",
)
