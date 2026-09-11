"""불변식 I1–I8의 실행 시 단언 (ARCHITECTURE.md §4).

아키텍처가 무너지면 실험 결과 전체가 무효가 되는 조건들이다. 각 불변식은
여기의 단언과 `tests/contracts/`의 테스트 양쪽으로 강제한다.

위반은 경고가 아니라 예외다 (CONVENTIONS.md §3).
"""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from typing import Any, Iterable, Iterator, Optional

import torch
import torch.nn as nn

from lsrr.core.errors import (
    CostAccountingError,
    FrozenBackboneViolation,
    GradientPathViolation,
    InjectionSpaceError,
    LeakageError,
    MultipleEncodeError,
    SharedReadoutViolation,
    TerminationFallbackMissing,
)

IGNORE_INDEX = -100
"""손실 타깃의 패딩 마스크 값. `labels`는 이 값으로, `target_ids`는 실제 토큰
id로 패딩된다 — 둘을 혼동하면 패딩을 학습한다 (I6, LEGACY_MAP.md §3)."""


# ------------------------------------------------------------------ I1


def freeze_module(module: nn.Module) -> nn.Module:
    """모듈을 동결하고 eval 모드로 고정한다 (I1)."""
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)
    return module


def weight_hash(module: nn.Module, num_tensors: Optional[int] = None) -> str:
    """가중치 해시. 학습 전후 대조로 I1을 검증한다.

    Args:
        num_tensors: 앞에서부터 몇 개의 텐서만 해싱할지. 대형 백본에서 전체
            해싱이 느릴 때 쓴다. None이면 전체.
    """
    hasher = hashlib.sha256()
    for i, (name, tensor) in enumerate(sorted(module.state_dict().items())):
        if num_tensors is not None and i >= num_tensors:
            break
        hasher.update(name.encode("utf-8"))
        hasher.update(tensor.detach().to("cpu", torch.float32).numpy().tobytes())
    return hasher.hexdigest()[:16]


def assert_frozen(module: nn.Module, what: str = "backbone") -> None:
    """학습 가능한 파라미터가 하나도 없음을 확인한다 (I1)."""
    trainable = [n for n, p in module.named_parameters() if p.requires_grad]
    if trainable:
        raise FrozenBackboneViolation(
            f"{what}에 학습 가능한 파라미터 {len(trainable)}개가 있다: "
            f"{trainable[:5]}{' ...' if len(trainable) > 5 else ''}. "
            f"제안서 §5는 백본 완전 동결(LoRA 불포함)을 명시한다."
        )


def assert_weights_unchanged(before: str, after: str, what: str = "backbone") -> None:
    """가중치 해시가 학습 전후 동일한지 확인한다 (I1)."""
    if before != after:
        raise FrozenBackboneViolation(
            f"{what} 가중치가 변했다 (before={before}, after={after})."
        )


# ------------------------------------------------------------------ I2


class EncodeCounter:
    """샘플 배치당 백본 인코딩 횟수를 추적한다 (I2).

    사이클 루프 안에서 백본에 재진입하면 Coconut형 반복 호출이 되어 본 연구의
    효율 주장이 사라진다. 재인코딩 외부 루프는 ADR-010의 명시적 예외 플래그
    하에서만 허용된다.
    """

    def __init__(self, allow_reencoding: bool = False) -> None:
        self.allow_reencoding = allow_reencoding
        self.count = 0
        self.total = 0

    def reset(self) -> None:
        """새 배치 시작. 스텝마다 호출한다."""
        self.count = 0

    def record(self) -> None:
        self.count += 1
        self.total += 1
        if self.count > 1 and not self.allow_reencoding:
            raise MultipleEncodeError(
                f"이 배치에서 백본 인코딩이 {self.count}회 발생했다. 제안서는 "
                f"트랜스포머 1회 호출을 효율 주장의 근거로 삼는다. 재인코딩 "
                f"외부 루프가 의도라면 experimental.reencoding_loop=true로 "
                f"명시하라 (ADR-010)."
            )

    def __repr__(self) -> str:
        return f"EncodeCounter(count={self.count}, total={self.total})"


@contextmanager
def encode_guard(counter: EncodeCounter) -> Iterator[EncodeCounter]:
    """배치 단위 인코딩 가드."""
    counter.reset()
    yield counter


# ------------------------------------------------------------------ I3


def assert_shared_readout(readout_calls: Iterable[Any]) -> None:
    """모든 사이클이 동일 판독 경로 인스턴스를 썼는지 확인한다 (I3).

    사이클별 헤드는 anytime 성질을 파괴한다 (제안서 §5).
    """
    ids = {id(obj) for obj in readout_calls if obj is not None}
    if len(ids) > 1:
        raise SharedReadoutViolation(
            f"판독 경로 인스턴스가 {len(ids)}개 쓰였다. 제안서 §5는 풀링·융합·"
            f"주입의 판독 경로가 전 사이클 공유여야 함을 명시한다(사이클별 헤드 금지)."
        )


# ------------------------------------------------------------------ I4


def assert_no_grad(module: nn.Module, what: str = "backbone") -> None:
    """역전파 후 그래디언트가 축적되지 않았는지 확인한다 (I4)."""
    with_grad = [n for n, p in module.named_parameters() if p.grad is not None]
    if with_grad:
        raise GradientPathViolation(
            f"{what}에 그래디언트가 축적되었다: {with_grad[:5]}. 역전파는 "
            f"h_fusion 주입 위치를 통해서만 흘러야 한다 (제안서 §4.4)."
        )


# ------------------------------------------------------------------ I5


def assert_has_fallback(rule: Any) -> None:
    """종료 규칙이 유효한 M_max 폴백을 갖는지 확인한다 (I5)."""
    m_max = getattr(rule, "m_max", None)
    if m_max is None or int(m_max) < 1:
        raise TerminationFallbackMissing(
            f"{type(rule).__name__}에 유효한 m_max가 없다(={m_max}). 제안서 §4.3은 "
            f"M_max 폴백을 필수 명세로 둔다 — 일부 상태가 진동하는 사례가 보고되어 있다."
        )


# ------------------------------------------------------------------ I6


def assert_no_answer_leakage(
    input_ids: torch.Tensor, answer_ids: torch.Tensor
) -> None:
    """문맥 인코딩 입력에 정답 토큰열이 포함되지 않았는지 확인한다 (I6).

    정답 토큰열이 질문 토큰열의 부분열로 나타나면 누출로 판정한다.
    """
    for b in range(input_ids.shape[0]):
        q = input_ids[b].tolist()
        a = [t for t in answer_ids[b].tolist() if t != IGNORE_INDEX]
        if not a or len(a) > len(q):
            continue
        for i in range(len(q) - len(a) + 1):
            if q[i : i + len(a)] == a:
                raise LeakageError(
                    f"샘플 {b}의 인코딩 입력에 정답 토큰열이 위치 {i}에 나타난다. "
                    f"문맥 인코딩에 들어가는 것은 질문뿐이다."
                )


def assert_labels_masked(labels: torch.Tensor, pad_token_id: int) -> None:
    """패딩이 IGNORE_INDEX로 마스킹되었는지 확인한다 (I6).

    패딩을 실제 토큰 id로 채운 텐서를 손실 타깃으로 쓰면 패딩을 학습한다.
    """
    if (labels == pad_token_id).any() and not (labels == IGNORE_INDEX).any():
        raise LeakageError(
            f"labels에 pad_token_id({pad_token_id})가 마스킹되지 않은 채 남아 있다. "
            f"손실 타깃은 IGNORE_INDEX({IGNORE_INDEX})로 패딩해야 한다 — "
            f"`target_ids`(디코더 입력)와 `labels`(손실 타깃)를 혼동하지 말 것."
        )


# ------------------------------------------------------------------ I7


def assert_cost_includes_backbone(report: Any) -> None:
    """비용 보고에 백본 항이 포함되었거나 플래그가 세워졌는지 확인한다 (I7)."""
    includes = getattr(report, "flops_includes_backbone", None)
    backbone = float(getattr(report, "flops_backbone", 0.0) or 0.0)
    if includes and backbone <= 0.0:
        raise CostAccountingError(
            "flops_includes_backbone=True인데 flops_backbone이 0이다. 백본 1회 "
            "순전파를 뺀 수치는 효율 비교를 무의미하게 만든다. 측정 불가라면 "
            "0으로 두지 말고 flops_includes_backbone=False로 명시하라."
        )


# ------------------------------------------------------------------ I8


def assert_injection_space(injected: torch.Tensor, d_in: int) -> None:
    """주입 벡터가 백본 입력 임베딩 공간에 있는지 확인한다 (I8).

    `[B, d_in]` 단일 벡터와 `[B, M, d_in]` 궤적 시퀀스를 모두 받는다 —
    v2.1 §4.4 는 M 개 잠재 사고 토큰을 방출하며, **그 모든 토큰**이 이 공간에
    있어야 한다 (ADR-015).

    어댑터 출력은 레이어 축 정렬을 위해 학습된 별개 공간이며, 이를 주입하면
    매니폴드 불일치가 발생한다 — 제안서 §9가 "원천 차단"을 기여로 내건 실패 양식이다.
    """
    if injected.dim() not in (2, 3):
        raise InjectionSpaceError(
            f"주입 벡터는 [B, d_in] 또는 [B, M, d_in]이어야 하는데 "
            f"{tuple(injected.shape)}이다."
        )
    if injected.shape[-1] != d_in:
        raise InjectionSpaceError(
            f"주입 벡터의 폭이 {injected.shape[-1]}인데 백본 입력 임베딩 폭은 "
            f"{d_in}이다. 융합 잔차 앵커는 어댑터 통과 전 백본 원본 h^(L)이어야 "
            f"하고 W_r의 출력 폭은 d_in이어야 한다 (ADR-003)."
        )
    if injected.dim() == 3 and injected.shape[1] < 1:
        raise InjectionSpaceError(
            f"궤적이 비어 있다: {tuple(injected.shape)}. 최소 1개 토큰을 방출해야 "
            f"한다 (M_min ≥ 1)."
        )


__all__ = (
    "IGNORE_INDEX",
    "freeze_module",
    "weight_hash",
    "assert_frozen",
    "assert_weights_unchanged",
    "EncodeCounter",
    "encode_guard",
    "assert_shared_readout",
    "assert_no_grad",
    "assert_has_fallback",
    "assert_no_answer_leakage",
    "assert_labels_masked",
    "assert_cost_includes_backbone",
    "assert_injection_space",
)
